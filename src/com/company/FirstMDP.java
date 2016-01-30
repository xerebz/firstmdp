package com.company;
import burlap.behavior.singleagent.ValueFunctionInitialization;
import burlap.behavior.singleagent.learning.actorcritic.ActorCritic;
import burlap.behavior.singleagent.learning.actorcritic.critics.TDLambda;
import burlap.behavior.singleagent.learning.actorcritic.actor.BoltzmannActor;
import burlap.behavior.statehashing.DiscreteStateHashFactory;
import burlap.domain.singleagent.graphdefined.GraphDefinedDomain;
import burlap.domain.singleagent.graphdefined.GraphTF;
import burlap.oomdp.auxiliary.DomainGenerator;
import burlap.oomdp.core.*;
import burlap.oomdp.singleagent.GroundedAction;
import burlap.oomdp.singleagent.RewardFunction;

public class FirstMDP {

    DomainGenerator dg;
    Domain domain;
    State initialState;
    RewardFunction rf;
    TerminalFunction tf;
    DiscreteStateHashFactory hashFactory;
    ValueFunctionInitialization vinit;

    public FirstMDP(double probToState, double[] valueEstimates, double[] rewards) {

        int numStates = 7;
        this.dg = new GraphDefinedDomain(numStates);

        ((GraphDefinedDomain) this.dg).setTransition(0, 0, 1, probToState);
        ((GraphDefinedDomain) this.dg).setTransition(0, 1, 2, 1. - probToState);
        ((GraphDefinedDomain) this.dg).setTransition(1, 0, 3, 1.);
        ((GraphDefinedDomain) this.dg).setTransition(2, 0, 3, 1.);
        ((GraphDefinedDomain) this.dg).setTransition(3, 0, 4, 1.);
        ((GraphDefinedDomain) this.dg).setTransition(4, 0, 5, 1.);
        ((GraphDefinedDomain) this.dg).setTransition(5, 0, 6, 1.);

        this.domain = this.dg.generateDomain();
        this.initialState = GraphDefinedDomain.getState(domain, 0);
        this.rf = new CustomRF(rewards);
        this.tf = new GraphTF(6);
        this.hashFactory = new DiscreteStateHashFactory();
        this.vinit = new CustomVFI(valueEstimates);

    }

    public double runTD(double gamma) {
        double learningRate = 1.0;
        double lambda = 1.0;
        TDLambda tdl = new TDLambda(rf,
                tf,
                gamma,
                hashFactory,
                learningRate,
                vinit,
                lambda);

        // A Boltzmann actor chooses actions at each state based on a particular probability
        // distribution. If there is just one possible action per state (as in this example),
        // it just takes the only action available to it.
        BoltzmannActor actor = new BoltzmannActor(this.domain, this.hashFactory, learningRate);

        ActorCritic ac = new ActorCritic(this.domain, this.rf, this.tf, gamma, actor, tdl);
        ac.runLearningEpisodeFrom(initialState);
        return tdl.value(initialState); // Returns the estimated value of the initial state based on
        // the learning episode. If there are stochastic actions,
        // only the outcomes actually observed will contribute to the
        // TD estimate.
    }

    public static class CustomRF implements RewardFunction {
        double[] rewards;
        public CustomRF(double[] rewards) {
            this.rewards = rewards;
        }

        @Override
        public double reward(State s, GroundedAction a, State sprime) {
            int sid = GraphDefinedDomain.getNodeId(s);
            return this.rewards[sid];
        }
    }

    public static class CustomVFI implements ValueFunctionInitialization {
        double[] valueEstimates;
        public CustomVFI(double[] valueEstimates) {
            this.valueEstimates = valueEstimates;
        }

        @Override
        public double value(State s) {
            int sid = GraphDefinedDomain.getNodeId(s);
            return this.valueEstimates[sid];
        }

        @Override
        public double qValue(State s, AbstractGroundedAction a) {
            int sid = GraphDefinedDomain.getNodeId(s);
            return this.valueEstimates[sid];
        }
    }

}
