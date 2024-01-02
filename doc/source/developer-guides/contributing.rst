How to contribute?
==================

The process of contributing to Qibochem largely follows Qibo itself, with the details given below.

Code review process
-------------------

All code submissions require a review and continuous integration tests before it can be merged to the main git branch.

We use the GitHub pull request mechanism which can be summarized as follows:

1. Fork the Qibochem repository.

2. Checkout main and create a new branch from it

    .. code-block::

        git checkout main -b new_branch

   where ``new_branch`` is the name of your new branch.

3. Implement your new feature on ``new_branch``.

4. After that, push your branch with:

    .. code-block::

        git push origin new_branch

5. At this point, you can create a pull request by visiting the Qibochem GitHub page.

6. The review process will start and changes in your code may be requested.

Tests
-----

When commits are pushed to the remote branches in the GitHub repository,
we perform integrity checks to ensure that the new code follows our coding conventions and does not break any existing functionality.

Any new changes must follow these code standards:

- **Tests**: We use pytest to run our tests that must be passed when new changes are integrated in the code. Regression tests, which are run by the continuous integration workflow are stored in ``qibochem/tests``. These tests contain several examples about how to use Qibochem.
- **Coverage**: Test coverage should be maintained at 100% when new features are implemented.
- **Pylint**: Test code for anomalies, such as bad coding practices, missing documentation, unused variables.
- **Pre commit**: We use pre-commit to enforce automation and to format the code. The `pre-commit ci <https://pre-commit.ci/>`_ will automatically run pre-commit whenever a commit is performed inside a pull request.

Besides the linter, further custom rules are applied e.g. checks for ``print`` statements that bypass the logging system
(such check can be excluded line by line with the ``# CodeText:skip`` flag).

Documentation
-------------

The Qibochem documentation is automatically generated with `sphinx <https://www.sphinx-doc.org/>`_,
thus all functions should be documented using docstrings.
The ``doc`` folder contains the project setup for the documentation web page.

The documentation requirements can be installed with:

.. code-block::

    pip install qibochem[docs]

In order to build the documentation web page locally please perform the following steps:

.. code-block::

    cd doc
    make html

The last command generates a web page in ``doc/build/html/``, which can be viewed by opening ``doc/build/html/index.html``.

The sections in the documentation are controlled by the ``*.rst`` files located in ``doc/source/``.


Jupyter notebooks
-----------------

Following qiboteam's `tutorial <https://github.com/qiboteam/tutorials/>`_ repository, add the following lines into your ``.git/config`` file.
This will clean all your notebook's output when you make a ``push``.

.. code-block::
    [filter "jupyter_clear_output"]
        clean = "jupyter nbconvert --stdin --stdout --log-level=ERROR \
                --to notebook --ClearOutputPreprocessor.enabled=True"
        smudge = cat
        required = true

    [core]
        attributesfile = .gitattributes
