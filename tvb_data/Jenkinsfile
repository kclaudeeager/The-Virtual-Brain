pipeline {
    agent {
        label 'docker_build'
    }

    environment {
        FULL_DOCKER_IMAGE_NAME = 'docker-repository.codemart.ro/tvb-build'
        LATEST_TAG = 'latest'
    }

    stages {

        stage ('Build ZIP of tvb-data for Zenodo') {
            steps {
                zip zipFile: 'tvb_data.zip', archive: 'true'
            }
        }

        stage ('Build Pypi package from tvb-data') {
            agent {
                docker {
                    image '${FULL_DOCKER_IMAGE_NAME}:${LATEST_TAG}'
                }
            }
            steps {
                sh '''#!/bin/bash
                      source activate tvb-run
                      mv MANIFEST.in MANIFEST.in_zenodo
                      mv MANIFEST.in_pypi MANIFEST.in
                      python setup.py sdist
                      python setup.py bdist_wheel
                      mv MANIFEST.in MANIFEST.in_pypi
                      mv MANIFEST.in_zenodo MANIFEST.in
                '''
                archiveArtifacts artifacts: 'dist/*, tvb_data/Default_Project.zip'
            }
        }
    }

    post {
        changed {
            mail to: 'lia.domide@codemart.ro',
            subject: "Jenkins Pipeline ${currentBuild.fullDisplayName} changed status",
            body: """
                Result: ${currentBuild.result}
                Job '${env.JOB_NAME} [${env.BUILD_NUMBER}]'
                Check console output at ${env.BUILD_URL}"""
        }

        success {
            echo 'Build finished successfully'
        }
    }
}