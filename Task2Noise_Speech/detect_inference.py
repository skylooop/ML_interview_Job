from absl import flags, app


FLAGS = flags.FLAGS

flags.DEFINE_string("mel_spec", default="", help="Path to mel spectrogram file.")

def main():
    pass



if __name__ == "__main__":
    app.run(main)