node("ml3") {
  docker.image('docker:latest').inside { 
    try {
      stage("stage1") {
        sh("echo makbomb DevOps lessons")
      }
      stage("checkout") {
        checkout scm 
      }
      stage("kafka start") {
          sh("cd kafka && pwd") 
          sh("cd kafka && chmod +x setup.sh")
          sh("cd kafka && ls && ./setup.sh")
          sh("cd kafka && docker-compose up --detach")
          sh("sleep 5")
          sh("echo Kafka cluster started")
        }
      /* spin up main part here */
      stage("start") {
        sh("cd ../")
        sh("chmod +x setup.sh && ./setup.sh")
        sh("docker-compose up --detach --force-recreate")
      }
      stage("test") {
        sh("cd tests && docker-compose up --force-recreate > tests")
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
}
