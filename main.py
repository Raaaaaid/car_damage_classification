import argparse
import car_damage as cd
import tensorflow as tf

'''
Command Line Interface application for the car_damage package.
'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./application_config.json', help='Path to config.json file')
    parser.add_argument('--debug', type=bool, default=False, help='Debugging flag for main')
    args = parser.parse_args()

    cd.init(args.config)
    debug = args.debug
    while True:
        try:
            img_fp = str(input("Image filepath: "))
            image = tf.io.read_file(img_fp)
            image = tf.io.decode_jpeg(image, channels=3)
            if debug:
                print(image)
            results = cd.run_classification(image)
            print(results)
        except KeyboardInterrupt:
            print('\n')
            print('Shutting down application')
            break