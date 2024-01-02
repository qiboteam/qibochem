Installation instructions
=========================

Supported operating systems
---------------------------

At the moment, Qibochem is only actively supported on Linux and MacOS systems.
In principle, it should be possible to install and run Qibochem on Windows, but with the caveat that the PySCF driver for obtaining molecular integrals is not available on Windows systems.

.. note::
      Qibochem is supported for Python >= 3.9.

..
  TODO: update further when package released on pypi

Installing from source
----------------------

Using ``pip install``:

.. code-block::

    git clone https://github.com/qiboteam/qibochem.git
    cd qibochem
    pip install .

Using ``poetry``:

.. code-block::

    git clone https://github.com/qiboteam/qibochem.git
    cd qibochem
    poetry install

For larger simulations, it may be desirable to install `other backend drivers <https://qibo.science/qibo/stable/getting-started/backends.html/>`_ for Qibo as well.
