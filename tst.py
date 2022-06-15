from argparse import ArgumentParser
import wandb
import random


def main(args):
    print(args.acc)


if __name__ == "__main__":
    # iterations = 30
    # destenation_dir = "data"
    # mri_ds = MRIDataset()
    # start = time.time()
    # for i in range(iterations):
    #     img_path, subject = mri_ds.get_path_and_subject(i)
    #     dest_path = os.path.join(destenation_dir, subject + ".npy")
    #     tmp = np.load(dest_path)
    # print(f"mean time = {(time.time() - start)/iterations}")

    # wandb.login(key='eb1e510a4bed996a9dac07bf3d3a2bda00cb113d')
    #
    # wandb.init(
    #     project="pytorch-intro",
    #     name="c",
    #     config={
    #         "epochs": 10,
    #         "batch_size": 128,
    #         "lr": 1e-3,
    #         "dropout": random.uniform(0.01, 0.80),
    #         "my_parameter": [2, 3, 4, 5],
    #         })
    # t()
    # print(wandb.config)
    # for i in range(1, 10):
    #     metrics = {"train/train_loss": i/2,
    #                "train/epoch": 2/i,
    #                "train/mae": 2*i}
    #     wandb.log(metrics)
    # val_metrics = {"val/val_loss": 22,
    #                "val/val_mae": 24}
    # wandb.log({**metrics, **val_metrics})
    # wandb.summary['test_accuracy'] = 0.8
    #
    # # ?? Close your wandb run
    # wandb.finish()

    parser = ArgumentParser()
    parser.add_argument("--GPU")
    parser.add_argument("--b")
    args = parser.parse_args()

    main(args)