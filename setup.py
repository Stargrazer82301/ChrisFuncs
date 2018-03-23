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
			  install_requires = ['astropy>=2.0',
                                     'astroquery>=0.3.7',
                                     'reproject>=0.4',
                                     'wget>=3.2'],
                 #dependency_links=['github.com/keflavich/FITS_tools/tarball/master#egg=FITS_tools-1.0'],
                 zip_safe = False)

