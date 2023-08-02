pipeline {
    agent any
    
    stages {
        stage('Build') {
            steps {
                sh 'docker build -t yamiannephilim/portugal-hotel-booking:latest .'
            }
        }

        stage('Clean') {
            steps {
                script {
                    def containerName = sh(returnStdout: true, script: 'docker ps -aqf "name=portugal-hotel-booking"').trim()
                    if (containerName) {
                        sh "docker stop $containerName"
                        sh "docker rm $containerName"
                    }
                }
            }
        }

        stage('Run') {
            steps {
                sh 'docker container stop portugal-hotel-booking || echo "this container does not exist"'
                sh 'docker network create yan || echo "this network exist"'
                sh 'echo y | docker container prune'
                sh 'docker run --name portugal-hotel-booking --network yan --restart=unless-stopped -p 8050:8050 -d yamiannephilim/portugal-hotel-booking:latest'
            }
        }
    }

    post {
        always {
            cleanWs()
        }
    }
}
