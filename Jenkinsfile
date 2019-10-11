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
        println "Branch name: "
        echo env.BRANCH_NAME
     }
      stage("Checkout version") {
        checkout scm
      }
      stage("Validate files integrity") {
	String ip = "kirill@" + getJenkinsMaster() + ":~/radioweb_jenkins/"
	
        sh("scp -r " + ip + "weights dnn_backend/") 
        sh("scp " + ip + "research.zip tests/code/")
	sh("cp tests/code/research.zip django/app/Slicer")
    	
        def str = sh(script: 'find dnn_backend/ -name "*pth.tar" | xargs du -hs | awk \'{print $1}\' | sed \'s/M//\' | awk \'$1 < 1 {print "FAILED"}; END {}\'', returnStdout: true)
	println str
        if (str != "") {
           println "Not all containers started"
	   exit 1
        }
        else {
           println "Main cluster started"
        }
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
          sh("docker-compose up --force-recreate --detach")
          sh("docker-compose logs --no-color > dnn_build.log")
          sh("sleep 20")
	  sh("cd tests && chmod +x check_build.sh && ./check_build.sh $BRANCH_NAME")    
          sh("echo main part started")
        }
      }
      stage("Test django module") { 
       dir("${WORKSPACE}") {
         sh("docker-compose exec -d django /bin/bash -c \"python manage.py test\"")
         sh("docker cp \$(docker-compose ps -q django):/app/tests/tests.xml .")
       }
      }
      stage("Test dnn backend") {
        dir("${WORKSPACE}/tests") {
          sh("docker-compose build")
          sh("docker-compose up --force-recreate")
	  sh("docker-compose logs --no-color > tests.log")
	  sh("docker cp \$(docker-compose ps -q tests):/app/code/dnn_tests.xml .")
	  sh("ls")
        }
      }
      stage("Archive artifacts") {
        archiveArtifacts("**/*.log*")
        archiveArtifacts("**/*tests*")
        junit("**/dnn_tests.xml")
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
