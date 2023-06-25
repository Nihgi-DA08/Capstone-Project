pipeline {
    agent any
    
    stages {
        stage('Build') {
            steps {
                sh 'docker build -t portugal-hotel-booking-dash:latest .'
            }
        }

        stage('Cleaning') {
            steps {
                script {
                    def containerName = sh(returnStdout: true, script: 'docker ps -aqf "name=portugal-hotel-booking-dash"').trim()
                    if (containerName) {
                        sh "docker stop $containerName"
                        sh "docker rm $containerName"
                    }
                }
            }
        }

        stage('Run') {
            steps {
                sh 'docker run --name portugal-hotel-booking-dash --network yan -d portugal-hotel-booking-dash:latest'
            }
        }
    }
}
