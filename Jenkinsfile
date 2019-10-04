def getJenkinsMaster() {
    return env.BUILD_URL.split('/')[2].split(':')[0]
}

node("ml2") {
    try {
      stage("Docker check") {
        sh("echo makbomb DevOps lessons")
        sh("docker version")
	println	"Master node: "
        println getJenkinsMaster()
      }
      stage("Checkout version") {
        checkout scm
      }
      stage("Validate files integrity") {
	String ip = "kirill@" + getJenkinsMaster() + ":~/radioweb_jenkins/"
	
        sh("scp -r " + ip + "weights dnn_backend/") 
        sh("scp " + ip + "research.zip tests/code/")
    	
	sh '''#!/bin/bash
	      echo "Checking size of weights files"
              str=$(find dnn_backend/ -name "*pth.tar" | xargs du -hs | awk '{print $1}' | sed 's/M//' | awk '$1 < 1 {print "FAILED"}; END {}')
              if [ -z "$str" ]; then 
		  echo "OK"
                  exit 0
              else
                  echo "Some files are < 1 MB"
                  exit 125
	      fi
	'''
      }
      stage("Kafka cluster start") {
	  dir("${WORKSPACE}/kafka") {
	     sh("pwd && ls")
	     sh("chmod +x setup.sh")
	     sh("./setup.sh")
	     sh("docker-compose up --detach")
	     sh("sleep 5")
	     sh("echo Kafka cluster started")
	  }
        }
      /* spin up main part here */
      stage("Main part start") {
        dir("${WORKSPACE}/") {
          sh("chmod +x setup.sh")
	  sh("./setup.sh")
          sh("docker-compose build")
          sh("docker-compose up --detach --force-recreate")
          sh("sleep 5")
          sh("echo main part started")
        }
      }
      stage("Test dnn backend") {
        dir("${WORKSPACE}/tests") {
          sh("docker-compose build")
          sh("docker-compose up --force-recreate")
	  sh("docker-compose logs --no-color >& log.tests")
        }
      }
      stage("Archive artifacts") {
        archiveArtifacts("**/*.tests*")
      }
    }
    catch(String error) {
      println(error)
      currentBuild.result = "FAILURE"
    } finally { 
       stage("Clear") {
       dir("${WORKSPACE}/kafka") { 
         sh("docker-compose down")
       }
       dir("${WORKSPACE}") {
         sh("docker-compose down")
       }
       dir("${WORKSPACE}/tests") {
         sh("docker-compose down")
       }
       cleanWs()
     }
   }
}
