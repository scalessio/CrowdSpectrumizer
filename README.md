# A Framework for Spectrum Classification using Crowdsensing Platforms
The repository contains the code and data of the paper "A Framework for Spectrum Classification using Crowdsensing Platforms" by
A. Scalingi, D. Giustiniano, R. Calvo-Palomino, N. Apostolakis and G. Bovet. Please cite the paper if you plan to use it in your publication.

## Abstract
We propose a framework that relies solely on Power Spectrum Density (PSD) data collected by low-cost RTL-SDR receivers.
This release contains the backend application of the framework developed with Flask which consist in two main packages.
First, the unsupervised transmission detection that works with PSD data already collected by the 
backend of the Electrosense platform and that provides stable detection of transmission boundaries. 
Second, the data-driven deep learning solution to classify the radio frequency communication 
technology used by the transmitter, using transmission features in a compressed space extracted from
single PSD measurements over at most 2 MHz band for inference.

We release the containerized version of the framework running as lightweight Docker container. 

## Dataset
We release the compressed version of the dataset that contains the measurements of 
real-world data collected from 47 different sensors deployed across Europe.
To access the dataset: https://kaggle.com/datasets/049e6c4449009037995ccdad1853a73bc9397b71ccf949d954955b58ddff3d79

## Requirements 
In order to run the repository you need the following software:
- Python 3
- Docker Engine
- Docker Compose

## Installation and Run
Install Docker Engine
https://docs.docker.com/engine/install/ubuntu/

Install Docker Compose https://docs.docker.com/compose/install/

After docker installation run the container with the following command `docker compose up`. 
The container will expose the port 5005.

### Test API
After run the container make sure that the backend is running: (1) Open the browser and (2) Type localhost:5005. 
If "App is Running" shows up then the backend is running correctly.  The backend expose the API to run the technology 
classification pipeline. Without the frontend, you can run the technology classification through curl:

`curl -i -H "Content-Type: application/json; charset=utf-8" -X POST -d '{"snsid" : "202481596708292", "snsname" : "rack_3", "month" : "May", "day" : "1" , "nation" : "Esp", "technology" : "test", "startf" : "20", "endf" : "1500", "freq_start":"20000000", "freq_end":"1500000000" }' http://localhost:5005/services/api/v1.0/usertc/pipeline
`



## License
```
Copyright (C) IMDEA Networks 2016

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see http://www.gnu.org/licenses/.
```

## Final notes
* The final dataset will be released in the next months because this repository is over its data quota. And we can not upload more data.
