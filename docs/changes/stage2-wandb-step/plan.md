# Stage-2 W&B Step axis

User instruction, preserved verbatim:

> The older w&b runs had "Step" as their x axis and now it is called "train/optimizer\_step" so I cant see new and old runs on the same plot, we need to fix it to just use "Step" (it should be optimizer setp)

This is the approved requirement. Implementation details and compatibility choices
are recorded separately in decisions.md. Existing recorded run histories are
append-only; changing an existing chart axis does not rewrite historical steps.
