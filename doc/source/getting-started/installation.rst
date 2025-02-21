Installation instructions
=========================

Supported operating systems
---------------------------

At the moment, Qibochem is only actively supported on Linux and MacOS systems.
In principle, it should be possible to install and run Qibochem on Windows as well (see below).

.. note::
      Qibochem is supported for Python >= 3.9.

Installing with pip
-------------------

The latest PyPI release of Qibochem can be installed directly with ``pip``:

.. code-block::

    pip install qibochem

The ``pip`` program will download and install all of the required dependencies for Qibochem.

.. note::
    For larger simulations, it may be desirable to install `other backend drivers <https://qibo.science/qibo/stable/getting-started/backends.html/>`_ for Qibo as well.

Installing from source
----------------------

The latest (development) version of Qibochem can be installed directly from the source repository on Github:

.. code-block::

    git clone https://github.com/qiboteam/qibochem.git
    cd qibochem
    pip install .

Installing on Windows
---------------------

The simplest way to get Qibochem on a Windows system is to use `Windows Subsystem for Linux <https://learn.microsoft.com/en-us/windows/wsl/install/>`_, and install Linux on Windows.
Qibochem can then be installed directly using ``pip`` as described above.
