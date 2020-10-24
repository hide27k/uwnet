---
layout: default
title: Test Space
---

## Neural Network Example
{% include chart
chart='
graph LR
    x1(("\(x_1\)")) -->|"\(w_1\)"| y1(("\(y_1\)"))
    x2(("\(x_2\)")) -->|"\(w_2\)"| y1
    x3(("\(x_3\)")) -->|"\(w_3\)"| y1
    x4(("\(x_4\)")) -->|"\(w_4\)"| y1
    x1 --> |"\(w_5\)"| y2(("\(y_1\)"))
    x2 --> |"\(w_6\)"| y2
    x3 --> |"\(w_7\)"| y2
    x4 --> |"\(w_8\)"| y2
    y1 --> |"\(v_1\)"| z(("\(z\)"))
    y2 --> |"\(v_2\)"| z
'
caption="An example, 2 layer neural network. The first layer is computed as \(y = wx\). The second layer is computed as \(z = vy\)"
%}

## Loss Functions
{% include chart
chart='
graph LR
    a["model parameters <br /> (weights)"] --> model
    subgraph Loss Function
    model --> | predictions | error
    data --> | labels | error
    data --> | features | model
    end
    error --> loss
'
caption="The loss function takes as input the parameters to the model. It uses those parameters to calculate predictions based on features from the data. It compares these predictions to the labels from that data and outputs the calculated loss."
%}
