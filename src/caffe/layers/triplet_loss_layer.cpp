#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layers/triplet_loss_layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void TripletLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  N = this->layer_param_.triplet_loss_param().group_size();
  margin = this->layer_param_.triplet_loss_param().margin();
  scale = this->layer_param_.triplet_loss_param().scale();
  LOG(INFO) << "Triplet loss bottom num is " << bottom[0]->num();
  LOG(INFO) << "Triplet loss group size is " << N;
  LOG(INFO) << "Triplet loss scale is " << scale;
  // batch size must be multiple times of N
  if(bottom[0]->num()%N!=0) {
      LOG(FATAL) << "batch size must be multiple times of group size: " << bottom[0]->num() << ", " << N;
  }
  // CHECK_EQ(bottom[0]->num()%N, 0);
  group_num = bottom[0]->num()/N;

  feat_len = bottom[0]->channels();

  CHECK_EQ(bottom.size(), 2);
  CHECK_EQ(bottom[0]->num(), bottom[1]->num());
  CHECK_EQ(bottom[0]->height(), 1);
  CHECK_EQ(bottom[0]->width(), 1);
  CHECK_EQ(bottom[1]->channels(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);

  diff_.Reshape(N*group_num, feat_len, 1, 1);
  // diff_sq_.Reshape(N*group_num, feat_len, 1, 1);
  dist_sq_.Reshape(N*group_num, 1, 1, 1);

  diff_ap_.Reshape(group_num, feat_len, 1, 1);
  diff_an_.Reshape(group_num, feat_len, 1, 1);
  diff_pn_.Reshape(group_num, feat_len, 1, 1);

  // diff_ap_sq_.Reshape(group_num, feat_len, 1, 1);
  // diff_an_sq_.Reshape(group_num, feat_len, 1, 1);

  dist_ap_sq_.Reshape(group_num, 1, 1, 1);
  dist_an_sq_.Reshape(group_num, 1, 1, 1);
}

template <typename Dtype>
void TripletLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->num(), N*group_num);
}

template <typename Dtype>
void TripletLossLayer<Dtype>::Forward_cpu(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    pos_ids = std::vector<std::vector<int> >(group_num, std::vector<int>());
    neg_ids = std::vector<std::vector<int> >(group_num, std::vector<int>());
    pn_ids = std::vector<std::pair<int, int> >(group_num, std::pair<int, int>(0, 0));
    const Dtype *feat_ptr = bottom[0]->cpu_data();
    const Dtype *label_ptr = bottom[1]->cpu_data();
    Dtype *diff_ptr_ = diff_.mutable_cpu_data();
    Dtype loss(0);

    caffe_set(feat_len*group_num, Dtype(0), diff_ap_.mutable_cpu_data());
    caffe_set(feat_len*group_num, Dtype(0), diff_an_.mutable_cpu_data());
    caffe_set(feat_len*group_num, Dtype(0), diff_pn_.mutable_cpu_data());

    Dtype cnt(0);
    for(int i=0; i<group_num; ++i) {
        for(int j=1; j<N; ++j) {
            if(label_ptr[i*N]==label_ptr[i*N+j]) pos_ids[i].push_back(j);
            else neg_ids[i].push_back(j);
            // f[0]-f[i]
            caffe_sub(feat_len, feat_ptr+feat_len*i*N, feat_ptr+feat_len*(i*N+j), diff_ptr_+feat_len*(i*N+j));
            if(scale!=1)
                caffe_cpu_scale(feat_len, scale, diff_ptr_+feat_len*(i*N+j), diff_ptr_+feat_len*(i*N+j));
            dist_sq_.mutable_cpu_data()[i*N+j] = caffe_cpu_dot(feat_len, diff_ptr_+feat_len*(i*N+j), diff_ptr_+feat_len*(i*N+j));
            // LOG(INFO) << "dist_sq_[" << j << "]" << dist_sq_.cpu_data()[i*N+j];
        }
        /* skip this group */
        if(pos_ids[i].size()==0 || neg_ids[i].size()==0) continue;
        // CHECK_GE(pos_ids[i].size(), 1);
        // CHECK_GE(neg_ids[i].size(), 1);
        int pos_max = 0, neg_min = 0;
        Dtype pos_max_val = 0, neg_min_val = 0;
        for(int ind=0; ind<pos_ids[i].size(); ind++) {
            int j = pos_ids[i][ind];
            CHECK_GE(j, 1);
            Dtype t = dist_sq_.cpu_data()[i*N+j];
            if(pos_max==0 || t>pos_max_val) {
                pos_max_val = t;
                pos_max = j;
            }
        }
        for(int ind=0; ind<neg_ids[i].size(); ind++) {
            int j = neg_ids[i][ind];
            CHECK_GE(j, 1);
            Dtype t = dist_sq_.cpu_data()[i*N+j];
            if(neg_min==0 || t<neg_min_val) {
                neg_min_val = t;
                neg_min = j;
            }
        }
        CHECK_GE(pos_max, 1);
        CHECK_GE(neg_min, 1);

        pn_ids[i] = std::pair<int, int>(pos_max, neg_min);

        Dtype mdist = std::max(pos_max_val-neg_min_val+margin, Dtype(0));
        loss += mdist;
        // if(i%40==0)
            // LOG(INFO) << "group addr" << i*N << ", pos_max_val " << pos_max_val << ", neg_min_val " << neg_min_val << ", margin " << margin;
        // LOG(INFO) << "pos_max " << pos_max << ", neg_min " << neg_min;

        /* cached for backward */
        if(mdist>0) {
            // diff_ptr[i*N+pos_max] -> diff_ap_[i]
            caffe_copy(feat_len, diff_ptr_+feat_len*(i*N+pos_max), diff_ap_.mutable_cpu_data()+i*feat_len);
            // diff_ptr[i*N+neg_min] -> diff_an_[i]
            caffe_copy(feat_len, diff_ptr_+feat_len*(i*N+neg_min), diff_an_.mutable_cpu_data()+i*feat_len);
            // feat_ptr_[i*N+pos_max]-feat_ptr_[i*N+neg_min] -> diff_pn_[i]
            caffe_sub(feat_len, feat_ptr+feat_len*(i*N+pos_max), feat_ptr+feat_len*(i*N+neg_min), diff_pn_.mutable_cpu_data()+i*feat_len);
            if(scale!=1)
                caffe_cpu_scale(feat_len, scale, diff_pn_.mutable_cpu_data()+feat_len*i, diff_pn_.mutable_cpu_data()+feat_len*i);
        }
        cnt += 1;
    }
    // LOG(INFO) << "cnt = " << cnt;
    loss = loss / cnt / Dtype(2);
    top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void TripletLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    /* loss_weight */
    const Dtype alpha = top[0]->cpu_diff()[0]/group_num;
    CHECK_EQ(feat_len, bottom[0]->channels());
    CHECK_EQ(N*group_num*feat_len, bottom[0]->count());
    if(propagate_down[0]) {
        Dtype *bottom_diff = bottom[0]->mutable_cpu_diff();
        caffe_set(N*group_num*feat_len, Dtype(0), bottom_diff);
        for(int i=0; i<group_num; ++i) {
            // diff for the anchor sample, index 0
            caffe_cpu_axpby(feat_len, -alpha, diff_pn_.cpu_data()+i*feat_len, Dtype(0), bottom_diff+feat_len*(i*N));
            // diff for the positive sample, index pos_max
            caffe_cpu_axpby(feat_len, -alpha, diff_ap_.cpu_data()+i*feat_len, Dtype(0), bottom_diff+feat_len*(i*N+pn_ids[i].first));
            // diff for the negative sample, index neg_min
            caffe_cpu_axpby(feat_len, alpha, diff_an_.cpu_data()+i*feat_len, Dtype(0), bottom_diff+feat_len*(i*N+pn_ids[i].second));
        }
    }
}

#ifdef CPU_ONLY
STUB_GPU(TripletLossLayer);
#endif
INSTANTIATE_CLASS(TripletLossLayer);
REGISTER_LAYER_CLASS(TripletLoss);

}
