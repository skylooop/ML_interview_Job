from absl import app, flags

# Instead of usual Argparse due to existing problems
FLAGS = flags.FLAGS

flags.DEFINE_string(
    "speech_dir",
    default="/home/m_bobrin/goznak/Task2Noise_Speech/Goznak_ML_Tasks/train1/train/train",
    help="Path to noisy and clean folders to train on.",
)
flags.DEFINE_boolean("num_epochs", default=1, help="Number of epochs of training.")


def main():
    """
    Entry point for classification task
    """


if __name__ == "__main__":
    app.run(main)
