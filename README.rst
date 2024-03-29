======
rlbook
======

.. image:: https://readthedocs.org/projects/rlbook/badge/?version=latest
        :target: https://rlbook.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status




Code for my walkthrough of:  
**Reinforcement Learning An Introduction by Richard Sutton and Andrew Barto** (http://incompleteideas.net/book/the-book.html)

* Free software: MIT license
* Documentation: https://rlbook.readthedocs.io.


Quickstart
--------

* Ch 2 Bandits (rlbook/bandits.py)
.. code-block:: bash

    cd experiments
    python bandits.py -m run.steps=1000 run.n_runs=2000 +bandit.epsilon=0.0,0.01,0.1
    aim up -p 6006 --repo outputs/bandit/


Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
