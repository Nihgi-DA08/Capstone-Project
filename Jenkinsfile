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
                sh 'docker stop portugal-hotel-booking-dash'
                sh 'docker rm portugal-hotel-booking-dash'
            }
        }

        stage('Run') {
            steps {
                sh 'docker run --name portugal-hotel-booking-dash --network yan -d portugal-hotel-booking-dash:latest'
            }
        }
    }
}
