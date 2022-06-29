# duplex2gif

## What it is?

This take scans of stereo pairs from [ISO Duplex 120](http://camera-wiki.org/wiki/Duplex_120) or [ISO Super Duplex 120](http://camera-wiki.org/wiki/Super_Duplex_120) and convert it to an animated GIF.

All cropping process done automatically, just keep borders black.

Camera:
![ISO Duplex Super 120](https://user-images.githubusercontent.com/5612507/175808901-55e88b37-06a4-4798-9cb5-5bff5f6a7eba.png)

What film look like:
![Film](https://user-images.githubusercontent.com/5612507/175809114-9c767728-e639-4894-be38-0091f52ad599.jpg)

Scan:
![Scan](https://user-images.githubusercontent.com/5612507/175808760-da9d27ac-8b00-4571-86d4-9b16a6e2261e.jpg)

Resulting GIF:
![GIF](https://user-images.githubusercontent.com/5612507/176485997-4e561ff3-61fb-4b87-b70d-a868b25db0f1.gif)

## How to use?

1. Clone or download repository.
2. Make sure you have [Python](https://www.python.org/) interpreter installed.
   ```shell
   python --version
   ```
3. Install requirement. Better to do this in [virtual environment](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/).
   ```shell
   pip install -r requirements.txt
   ```
4. Run script.

   Like this if you copy images to script directory.
   ```shell
   python duplex2gif.py
   ```
   or explicitly specified directory with scans.
   ```shell
   python duplex2gif.py -p <path_to_images>
   ```
   
## TODO

- [x] Basic cropping and merging
- [x] Check on B&W film
- [x] Implement more robust cropping algorithm
- [ ] Check on color film
- [ ] Add process params to CLI
