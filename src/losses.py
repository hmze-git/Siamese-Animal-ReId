from pytorch_metric_learning import losses

def get_loss(loss_name="arcface", embedding_dim=512, num_classes=1000, margin=0.5, scale=64):
    if loss_name == "arcface":
        #this one usually works best for this kind of stuff
        return losses.ArcFaceLoss(num_classes=num_classes, embedding_size=embedding_dim, margin=margin, scale=scale)
    elif loss_name == "cosface":
        return losses.CosFaceLoss(num_classes=num_classes, embedding_size=embedding_dim, margin=margin, scale=scale)
    elif loss_name == "triplet":
        return losses.TripletMarginLoss(margin=margin)
    else:
        raise ValueError(f"unknown loss: {loss_name}")