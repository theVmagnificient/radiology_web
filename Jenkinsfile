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
         stage("Test django") {
	         dir("${WORKSPACE}") {
            sh("ls tests/code/")
            sh("docker cp tests/code/research.zip \$(docker-compose ps -q django):/app/tests/research123.zip")
            sh("docker-compose exec -d django /bin/bash -c \"python manage.py test\"")
            sleep 30
            sh("docker cp \$(docker-compose ps -q django):/app/tests/tests.xml . && mv tests.xml django_tests.xml")
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
  buildStages.put("Telegram bot & Auth service tests", {
          stage("Test tg bot") {
            dir("${WORKSPACE}/go_services") {
              println "Putting images for tests into the aimed_results volume" 
              sh("docker container create --name temp -v aimed_results:/data busybox")
              sh("docker cp bot_aimed/res1 temp:/data")
              sh("docker-compose -f docker-compose.test.yml build")
              sh("docker-compose -f docker-compose.test.yml up -d db_auth")
              sh("docker-compose -f docker-compose.test.yml up auth bot flask")
              sh("docker-compose logs --no-color > tg_bot.log")
              sh("docker cp \$(docker-compose -f docker-compose.test.yml ps -q bot):/src/bot_tests.xml .")
              sh("docker cp \$(docker-compose -f docker-compose.test.yml ps -q auth):/src/auth_tests.xml .")
              sh("docker-compose -f docker-compose.test.yml down")
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

        sh("scp " + ip + "auth_service/db.secret go_services/auth_aimed/src")
        sh("scp " + ip + "auth_service/init.sql go_services/auth_aimed/database")

        sh("scp " + ip + "bot_tg/token.secret go_services/bot_aimed/src")
        sh("scp " + ip + "bot_tg/test/* go_services/flask_test_pyrogram/")
        sh("scp -r " + ip + "res1 go_services/bot_aimed/")

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
    }
    catch(String error) {
      println(error)
      currentBuild.result = "FAILURE"
    } finally { 
       stage("Archive artifacts") {
         archiveArtifacts("**/*.log*")
         archiveArtifacts("**/*tests*")
         junit("**/*tests.xml")
       }
       stage("Clear") {
        sh("docker rm temp")
        dir("${WORKSPACE}/kafka") { 
          sh("docker-compose down")
        }
        dir("${WORKSPACE}") {
          sh("docker-compose down")
          sh("chmod +x uninstall.sh")
          sh("./uninstall.sh")
        }
        dir("${WORKSPACE}/tests") {
          sh("docker-compose down")
        }
        cleanWs()
     }
   }
}
