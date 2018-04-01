## ChrisFuncs

My far-too-large module of various astronomical/statistical/miscellaneous convenience functions. 

### Installation

1. [Download](https://github.com/Stargrazer82301/ChrisFuncs/archive/master.zip) the zip archive containing ChrisFuncs from the GitHub repository, and extract it to a temporary location of your choice.
2. Open a terminal and navigate to extracted directory that contains setup.py
3. Install using the command `pip install -e . --user` (Whilst you can try installing using the more common `python setup.py install` route, this sometimes hits compatibility issues, whereas the pip local installer seems more robust.)

### Usage

ChrisFuncs is imported as:
```
import ChrisFuncs
```
Functions and sub-modules can be found therein. For example, the `SigmaClip` function is called via `ChrisFuncs.SigmaClip(*args)`. Whilst the `Photom` sub-module is imported via `import ChrisFuncs.Photom`.

Enjoy!
