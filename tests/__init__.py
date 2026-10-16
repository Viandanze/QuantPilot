"""Package marker for the test suite.

Making ``tests/`` a regular package is deliberate: some third-party wheels
(for example ultralytics/roboflow) install their own top-level ``tests``
package into site-packages, and Python resolves a regular package there over
any same-named *namespace* portion earlier on sys.path - which used to hijack
``from tests.conftest import ...`` in this suite and break collection on
machines that have those wheels installed. With this ``__init__.py`` present,
the project-local package wins by sys.path order instead.
"""
