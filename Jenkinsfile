node("ml3") {
  try {
    stage("stage1") {
      sh("echo makbomb DevOps lessons")
    }
    stage("checkout") {
      checkout scm 
    }
    stage("test") {
      sh("cd tests && docker-compose up")
    }
  }
  catch(String error) {
    println(error)
    currentBuild.result = "FAILURE"
  }
  stage("clear") {
    cleanWs() 
  }
}
