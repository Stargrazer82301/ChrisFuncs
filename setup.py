# Import smorgasbord
import setuptools

# Configure setup
setuptools.setup(name = 'chrisfuncs',
                 version = '1.0',
                 description = 'My far-to-large package of various astro/stats/misc convenience functions.',
                 url = 'https://github.com/Stargrazer82301/ChrisFuncs',
                 author = 'Chris Clark (github.com/Stargrazer82301)',
                 author_email = 'cjrc88@gmail.com',
                 license = 'MIT',
                 classifiers = ['Programming Language :: Python :: 2.7',
                                'License :: OSI Approved :: MIT License'],
                 packages = setuptools.find_packages(),
                 zip_safe = False)
