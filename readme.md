##Radiology web service for lung nodules detection.

Deployment algorithm:

```

1) git clone https://github.com/dit-shared/radiology_web.git 
2) cd radiology_web 
3) chmod +x setup.sh
4) ./setup.sh   // creates neccesary volumes
5) cd kafka 
6) docker-compose up --detach   // starts kafka services
7) cd ../
8) docker-compose up --detach   // starts main main part

```
