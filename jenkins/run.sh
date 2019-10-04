docker pull jenkins/jenkins

docker run -p 10228:8080 -p 50000:50000 -d --network=kafka-network jenkins/jenkins:lts 
