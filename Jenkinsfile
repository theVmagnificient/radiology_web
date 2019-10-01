node("ml3") {
  docker.image('docker:latest').inside { 
    try {
      stage("stage1") {
        sh("echo makbomb DevOps lessons")
      }
      stage("checkout") {
        checkout scm 
      }
      /* spin up main part here */
      stage("start") {
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
