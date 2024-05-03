from glob import glob

def verify_dataset():
    """
    Checks if the dataset is valid:
    1. Verify that the number of items in each subset is as expected.
    2. Ensure that labels and original images in each subset match each other.
    3. Arrange dataset folder tree as follow.

        dataset tree:
            |- train
                |- *.lab_png    # label
                |- *.jpg        # orignal
            |- val 
                |- *.lab_png
                |- *.jpg
            |- test
                |- *.lab_png
                |- *.jpg

    """
    
    # Paths for training, validation, and test sets
    dataset_dir = "../autodl-tmp"
    
    train_lab = glob(f"{dataset_dir}/train/*.png")
    train_org = glob(f"{dataset_dir}/train/*.jpg")
    val_lab = glob(f"{dataset_dir}/val/*.png")
    val_org = glob(f"{dataset_dir}/val/*.jpg")
    test_lab = glob(f"{dataset_dir}/test/*.png")
    test_org = glob(f"{dataset_dir}/test/*.jpg")


    # Verify the number of items in each set
    assert len(train_lab) == len(train_org) == 1445, "Training set sizes do not match."
    assert len(val_lab) == len(val_org) == 450, "Validation set sizes do not match."
    assert len(test_lab) == len(test_org) == 448, "Test set sizes do not match."

    # Verify that labels and original images match
    assert set([t.split('/')[-1].split('_')[0] for t in train_lab]) == \
           set([t.split('/')[-1].split('.')[0] for t in train_org]), "Training labels and images do not match."
    assert set([t.split('/')[-1].split('_')[0] for t in val_lab]) == \
           set([t.split('/')[-1].split('.')[0] for t in val_org]), "Validation labels and images do not match."
    assert set([t.split('/')[-1].split('_')[0] for t in test_lab]) == \
           set([t.split('/')[-1].split('.')[0] for t in test_org]), "Test labels and images do not match."

    print("Dataset verification passed!")


if __name__ == "__main__":
    verify_dataset()
