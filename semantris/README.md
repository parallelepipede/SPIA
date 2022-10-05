<div align="center">

  <h3 align="center">Semantris Automation</h3>
</div>


<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#technical-stack">Technical Stack</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About the project

`Semantris` project consists of an autonomous bot playing Google game `Semantris` mixing semantics and Tetris.

![animated demo screenshot](https://github.com/parallelepipede/SPIA/tree/main/semantris/images/demo.gif)

Start a new game and let Machine Learning play for you. Compare the result with your best own high scores.

Semantris game provides a stack of various words. One of them is the target. The player enters a word as close as possible to the target. Then words in the stack are ordered based on semantics association.<br>

Try playing with slang, technical terms, pop culture references, synonyms, antonyms, and even full sentences.
Play game tutorial for a better understanding.

Have Fun !!

<!-- GETTING STARTED -->
## Getting Started
### Prerequisites
Jupyter-notebook and Python > 3.6 installed.<br>
The project is tested under Linux, MacOS. Windows compatibility is ensured.<br>

### Installation

Clone Semantris project : 

```sh
   git clone https://github.com/parallelepipede/SPIA.git
   cd SPIA/semantris
   ```

The project is based on jupyter-notebook files based on your base Python environment. Please make sure it doesn't impact other projects requirements.

Please follow pytesseract installation guide [pytesseract install](https://pypi.org/project/pytesseract/).

Make sure opencv-contrib-python is removed or run in terminal and install necessary libraries:
```sh
   pip uninstall opencv-contrib-python
   pip install -r requirements.txt
   ```

<!-- USAGE EXAMPLES -->
## Usage

`Semantris.ipynb` correspond to main gameplay.
`Annoyed.ipynb` corresponds to the conception of a new vocabulary used.

Run :
```sh
   jupyter-notebook semantris.ipynb
   ```

You need to skip tutorial when running the algorithm.<br>
Run all cells with jupyter-notebook and Semantris windows split as demonstrated in images for the gameplay.


<!-- Technical Stack -->
## Technical Stack

This part carefully explains technical parts of the algorithm.
`Semantris` Google source code provides no clear information about targeted word. The bot has to analyse the screen as a human. 

First operation is to spot the targeted word with the arrow. Then read the correspondant word and provide a word with close semantics. 



#### 1. Template Matching
First of all, the bot takes a screenshot of the current game.
The game image is grayscaled to reduce image computation. <br>
Template matching is based on opencv. It is used to locate the arrow template image in the screenshot.
It simply slides the template image over the input image as a 2D convolution.

TM_CCORR is the empirical selected method. Documentation can be found [here](https://docs.opencv.org/4.x/df/dfb/group__imgproc__object.html#gga3a7850640f1fe1f58fe91a2d7583695da5be00b45a4d99b5e42625b4400bfde65).

#### 2. OCR
Once the arrow template is spotted. A bounded box is declared around the word.<br>
Pytesseract is a powerful Optical Character Recognition algorithm, a wrapper of Tesseract.

#### 3. Sentence encoder

`Semantris` Google AI was trained on conversationnal text spanning a large variety of topics. It is a able to make many types of associations.

Natural Language Processing (NLP) is based on embeddings. Each word / sentence is converted into vectors. This is operated by the [universal sentence encoder](https://tfhub.dev/google/universal-sentence-encoder/4).

Please read [this article](https://www.tensorflow.org/text/guide/word_embeddings) for a first understanding of embeddings.

The universal sentence encoder embedding results in 512 dimension. Dimension is reduced to 64 based on [Gaussian Random Reduction projection](https://scikit-learn.org/stable/modules/generated/sklearn.random_projection.GaussianRandomProjection.html) to ease computation for next step.

#### 4. Annoy

The universal sentence encoder is used to generate embeddings. The next step is to select a vocabulary to learn and try to establish links, measures and computations on embeddings.

A list of common English words is selected from [this website](https://github.com/dwyl/english-words). Words are filtered and cleaned.
Embeddings for all English words are computed and dimension reduced.

As used by Spotify for users/items recommendation, [Annoy](https://github.com/spotify/annoy) (Approximate Nearest Neighbors Oh Yeah) search for points in space that are closed to a given points. Each group of word or word is represented with a point in 64 dimensions.

Nearest neighbors are computed for dimension reduced embeddings of targeted word.

#### 5. Autonomous Bot
The bot is generated using Pyautogui. It takes screenshots and automatically copy paste generated proposed word.

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->
## Contact

Paul Zanoncelli  - paul.zanoncelli@gmail.com

Project Link: [https://github.com/parallelepipede/SPIA](https://github.com/parallelepipede/SPIA)

<p align="right">(<a href="#readme-top">back to top</a>)</p>
