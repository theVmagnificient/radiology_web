def getJenkinsMaster() {
    return env.BUILD_URL.split('/')[2].split(':')[0]
}

// Create List of build stages to suit
def prepareBuildStages() {
  echo "Preparing docker stages"
  def buildList = []

  def buildStages = [:]

  buildStages.put("Kafka docker", {
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
    }
   )

  buildStages.put("Main docker", {
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
      }
    )


  buildList.add(buildStages)

  return buildList
}

// Create list of test stages 
def prepareTestStages() {
  echo "Preparing test stages"
  def buildList = []

  def buildStages = [:]

  buildStages.put("Django test", {
	      stage("Test dnn backend") {
		      dir("${WORKSPACE}/tests") {
		        sh("docker-compose build")
            sh("docker-compose up --force-recreate")
            sh("docker-compose logs --no-color > tests.log")
            sh("docker cp \$(docker-compose ps -q tests):/app/code/dnn_tests.xml .")
          }
	      }
      })
  buildStages.put("DNN test", {
          stage("Test dnn backend") {
            dir("${WORKSPACE}/tests") {
              sh("docker-compose build")
              sh("docker-compose up --force-recreate")
              sh("docker-compose logs --no-color > tests.log")
              sh("docker cp \$(docker-compose ps -q tests):/app/code/dnn_tests.xml .")
            }
          }
      })

  buildList.add(buildStages)

  return buildList
}

def buildStages 
def testStages
def runParallel = true

node("ml2") {
    try {
      stage('Initialise') {
        // Set up List<Map<String,Closure>> describing the builds
        buildStages = prepareBuildStages()
        testStages = prepareTestStages()
        println("Initialised pipeline.")
     }
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
     /* stage("Kafka cluster start") {
	     dir("${WORKSPACE}/kafka") {
	       sh("pwd && ls")
	       sh("chmod +x setup.sh")
	       sh("./setup.sh")
	       sh("docker-compose up --detach")
	       sh("sleep 5")
	       sh("echo Kafka cluster started")
	      }
      } */
      /* spin up main part here */
      /*
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
      } */
      for (builds in buildStages) {
        if (runParallel) {
          parallel(builds)
      } else {
          // run serially (nb. Map is unordered! )
          for (build in builds.values()) {
             build.call()
          }
        } 
      }
      for (tests in testStages) {
        if (runParallel) {
          parallel(tests)
        } else {
          for (test in test.values()) {
            test.call()
          }
        }
      }
//      parallel {
/*	      stage("Test django module") { 
	       dir("${WORKSPACE}") {
		 sh("ls tests/code/")
		 sh("docker cp tests/code/research.zip \$(docker-compose ps -q django):/app/tests/")
		 sh("docker-compose exec -d django /bin/bash -c \"python manage.py test\"")
                 sleep 30
		 sh("docker cp \$(docker-compose ps -q django):/app/tests/tests.xml . && mv tests.xml django_tests.xml")
	       }
	      }
	      stage("Test dnn backend") {
		dir("${WORKSPACE}/tests") {
		  sh("docker-compose build")
		  sh("docker-compose up --force-recreate")
		  sh("docker-compose logs --no-color > tests.log")
		  sh("docker cp \$(docker-compose ps -q tests):/app/code/dnn_tests.xml .")
		}
	      } */
 //     }
      stage("Archive artifacts") {
        archiveArtifacts("**/*.log*")
        archiveArtifacts("**/*tests*")
        junit("**/*tests.xml")
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
