# Registry and method contract

The registry resolves a configuration's method name to a `BaseMethod`
subclass. `BaseMethod` defines the lifecycle used by both training and
evaluation.

::: methods
    options:
      members:
        - get_method
        - list_methods
        - get_backbone_contracts

::: methods.base
    options:
      members:
        - BaseMethod
