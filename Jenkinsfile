node("ml2") {
    try {
      stage("docker check") {
        sh("echo makbomb DevOps lessons")
        sh("docker version")
      }
      stage("checkout") {
        checkout scm 
      }
      stage("kafka start") {
	  dir("${WORKSPACE}/kafka") {
	     sh("chmod +x setup.sh")
	     sh(". setup.sh")
	     sh("docker-compose up --detach")
	     sh("sleep 5")
	     sh("echo Kafka cluster started")
	  }
        }
      /* spin up main part here */
      stage("start") {
        dir("${WORKSPACE}/") {
          sh("chmod +x setup.sh && . setup.sh")
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
