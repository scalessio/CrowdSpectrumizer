#!/usr/bin/env bash

curl -i -H "Content-Type: application/json; charset=utf-8" -X POST -d '{"snsid" : "202481591141168", "snsname" : "Scalessio", "month" : "Jun", "day" : "14" , "nation" : "Ita", "technology" : "test", "startf" : "20", "endf" : "1500", "freq_start":"20000000", "freq_end":"1500000000" }' http://localhost:5005/services/api/v1.0/usertc/test
sleep 4
curl -i -H "Content-Type: application/json; charset=utf-8" -X POST -d '{"snsid" : "202481598508037", "snsname" : "dipolkurz", "month" : "May", "day" : "1" , "nation" : "Esp", "technology" : "test", "startf" : "20", "endf" : "1500", "freq_start":"20000000", "freq_end":"1500000000" }' http://localhost:5005/services/api/v1.0/usertc/test
sleep 4
curl -i -H "Content-Type: application/json; charset=utf-8" -X POST -d '{"snsid" : "202481597776599", "snsname" : "bcn-L", "month" : "Jun", "day" : "10" , "nation" : "Esp", "technology" : "test", "startf" : "20", "endf" : "1500", "freq_start":"20000000", "freq_end":"1500000000" }' http://localhost:5005/services/api/v1.0/usertc/test