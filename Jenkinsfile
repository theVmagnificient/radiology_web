def getJenkinsMaster() {
    return env.BUILD_URL.split('/')[2].split(':')[0]
}

node("ml2") {
    try {
      stage("docker check") {
        sh("echo makbomb DevOps lessons")
        sh("docker version")
	println	"Master node: "
        println getJenkinsMaster()
      }
      stage("checkout") {
        checkout scm
      }
      stage("validation") {
	String ip = "kirill@" + getJenkinsMaster() + ":~/weights"
	
        sh("scp -r " + ip + " dnn_backend/") 
    	
	sh '''#!/bin/bash
		echo "Checking size of weights files"
		str=$(find dnn_backend/ -name "*pth.tar" | xargs du -hs | awk '{print $1}' | sed 's/M//' | awk '$1 < 1 {print "FAILED"}; END {}')                                                                                                                                                                                                                       if [ -z "$str" ]; then                                                                                                                                                        echo "OK"                                                                                                                                                                   exit 0                                                                                                                                                                    else                                                                                                                                                                          exit 125                                                                                                                                                                  fi           
	'''
      }
      stage("kafka start") {
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
      stage("start") {
        dir("${WORKSPACE}/") {
          sh("chmod +x setup.sh")
	  sh("./setup.sh")
          sh("docker-compose up --detach --force-recreate")
          sh("sleep 5")
          sh("echo main part started")
        }
      }
      stage("test") {
        dir("${WORKSPACE}/tests") {
          sh("docker-compose up --force-recreate > tests")
        }
      }
      stage("archive") {
        archiveArtifacts("tests")
      }
    }
    catch(String error) {
      println(error)
      currentBuild.result = "FAILURE"
    }
    stage("clear") {
      cleanWs()
      sh("docker-compose rm tests")
    }
}
