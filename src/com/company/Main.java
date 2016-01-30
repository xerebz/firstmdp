package com.company;

public class Main {
    public static void main(String[] args) {
        double probToState = 0.81;
        double valueEstimates[] = {0.0,4.0,25.7,0.0,20.1,12.2,0.0};
        double rewards[] = {7.9,-5.1,2.5,-7.2,9.0,0.0,1.6};
        FirstMDP fmdp = new FirstMDP(probToState, valueEstimates, rewards);
        double gamma = 1.0;
        System.out.println(fmdp.runTD(gamma));
    }
}
