# A Framework for Spectrum Classification using Crowdsensing Platforms
The repository contains the code and dataset link of the paper "A Framework for Spectrum Classification using Crowdsensing Platforms" by
A. Scalingi, D. Giustiniano, R. Calvo-Palomino, N. Apostolakis and G. Bovet. Please cite the paper if you plan to use it in your publication.

## Abstract
We propose a framework that relies solely on Power Spectrum Density (PSD) data collected by low-cost RTL-SDR receivers.
This release contains the backend application of the framework developed with Flask which consist in two main packages.
First, the unsupervised transmission detection that works with PSD data already collected by the 
backend of the Electrosense platform and that provides stable detection of transmission boundaries. 
Second, the data-driven deep learning solution to classify the radio frequency communication 
technology used by the transmitter, using transmission features in a compressed space extracted from
single PSD measurements over at most 2 MHz band for inference.

We release the lightweight version of the framework as Docker container. 

## Dataset

We release the compressed version of the dataset that contains the measurements of 
real-world data collected from 47 different sensors deployed across Europe.

To access to the dataset: https://zenodo.org/record/7521246

If you encounter any issue with Zenodo please write an email to:
alessio.scalingi@imdea.org with subject "Access PSD Dataset - [Your Name]"

DOI: 10.5281/zenodo.7521246.

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
classification pipeline. Without the frontend, test the technology classification through curl with:

`curl -i -H "Content-Type: application/json; charset=utf-8" -X POST -d '{"snsid" : "202481596708292", "snsname" : "rack_3", "month" : "May", "day" : "1" , "nation" : "Esp", "technology" : "test", "startf" : "20", "endf" : "1500", "freq_start":"20000000", "freq_end":"1500000000" }' http://localhost:5005/services/api/v1.0/usertc/pipeline
`



## License
```
BSD 3-Clause License

Copyright (c) 2023, Alessio Scalingi, All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that 
the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions, and the 
following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions, and 
the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote 
products derived from this software without specific prior written permission.

4. When using this software, you are required to cite the following paper:

    A Framework for Spectrum Classification using Crowdsensing Platforms
    A. Scalingi, D. Giustiniano, R. Calvo-Palomino, N. Apostolakis and G. Bovet
    2023
    DOI 10.1109/INFOCOM53939.2023.10228867

    @INPROCEEDINGS{10228867,
    author={Scalingi, Alessio and Giustiniano, Domenico and Calvo-Palomino, Roberto and Apostolakis,
     Nikolaos and Bovet, Gérôme},
    booktitle={IEEE INFOCOM 2023 - IEEE Conference on Computer Communications}, 
    title={A Framework for Wireless Technology Classification using Crowdsensing Platforms}, 
    year={2023},
    volume={},
    number={},
    pages={1-10},
    doi={10.1109/INFOCOM53939.2023.10228867}
    url=}

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, 
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE 
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE 
USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

```
