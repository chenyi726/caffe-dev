#ifndef CAFFE_TRIPLET_LOSS_LAYER_HPP_
#define CAFFE_TRIPLET_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {
template <typename Dtype>
class TripletLossLayer : public LossLayer<Dtype> {
    public:
        explicit TripletLossLayer(const LayerParameter& param)
            : LossLayer<Dtype>(param) {}
        virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                const vector<Blob<Dtype>*>& top);
        virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                const vector<Blob<Dtype>*>& top);

        virtual inline const char* type() const { return "TripletLoss"; }
        virtual inline int ExactNumTopBlobs() const { return 1; }
        virtual inline int ExactNumBottomBlobs() const { return 2; }

    protected:
        virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                const vector<Blob<Dtype>*>& top);
        virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

        // probe id and gallery ids
        std::vector<std::vector<int> > pos_ids, neg_ids;
        std::vector<std::pair<int, int> > pn_ids;
        Dtype margin;
        Dtype scale;
        int N;
        int group_num;
        int feat_len;

        Blob<Dtype> diff_;
        // Blob<Dtype> diff_sq_;
        Blob<Dtype> dist_sq_;

        Blob<Dtype> diff_ap_;
        Blob<Dtype> diff_an_;
        Blob<Dtype> diff_pn_;
        // Blob<Dtype> diff_ap_sq_;
        // Blob<Dtype> diff_an_sq_;
        Blob<Dtype> dist_ap_sq_;
        Blob<Dtype> dist_an_sq_;
        int max_dist_id;
};
}

#endif
