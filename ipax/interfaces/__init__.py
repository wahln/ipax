# Copyright 2026 Niklas Wahl
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Front ends speaking another library's calling convention.

Each submodule is an application-edge adapter (allowed to import its concrete
library, like ``backend/sparse/``) that translates onto the public
:class:`~ipax.Problem` / :func:`~ipax.solve` API. Nothing here is imported by
``ipax`` itself, so ``import ipax`` stays free of concrete array libraries.

- :mod:`ipax.interfaces.scipy` — :func:`~ipax.interfaces.scipy.minimize_ipax`,
  usable standalone or as ``scipy.optimize.minimize(..., method=minimize_ipax)``.
"""
