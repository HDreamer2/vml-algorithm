# Project Name
Visible Financial Computing

## About

API: Receives HTTP requests from the Java backend and processes them using Flask framework.

Algorithm Module: Executes specific algorithms (e.g., linear regression, decision trees) based on the requests received from the Java backend. This module will be introduced in detail later.

### Key Features

- **Linear Regression**
- **Logistic Regression**
- **Random Forest**
- **Decision Tree**


## Installation Guide

### Prerequisites

Before you begin installation, ensure you have the following software or tools installed:
- docker

### Installation Steps

1. **build image**

docker build -t algorithm:latest -f Dockerfile

2. **run container**

docker run -d -p 8084:8084 --name algorithm algorithm:latest


