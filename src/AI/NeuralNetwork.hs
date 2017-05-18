{-# LANGUAGE ParallelListComp #-}
{-# LANGUAGE ViewPatterns #-}
{-# OPTIONS_GHC -Wno-type-defaults #-}

module AI.NeuralNetwork
    ( module AI.NeuralNetwork.ActivationFunctions
    , Neuron, Layer, Network
    , TrainingSample, TrainingSet
    , network
    , output
    , trainUntil, trainUntil'
    , trainOnce, trainOnce'
    ) where

import AI.NeuralNetwork.ActivationFunctions
import Control.Monad
import System.Random

type Neuron = ([Double], Double)
type Layer = [Neuron]
type Network = [Layer]
type TrainingSample = ([Double], [Double])
type TrainingSet = [TrainingSample]

network :: Int -> [Int] -> Int -> IO Network
network ils hls ols = gen ils $ hls ++ [ols]
  where gen ils' (l:ls) = (:) <$> (replicateM l $ (,) <$> replicateM ils' r <*> r) <*> gen l ls
        gen _ _ = return []
        r = randomRIO (-0.1, 0.1)

checkNet :: Network -> Network
checkNet net
    | length net >= 2 = net
    | otherwise = error "Neural network should have at least 2 layers, including inputs"

netInputs :: [Double] -> Layer -> [Double]
netInputs is = map $ \(ws, b) -> sum (zipWith (*) is ws) + b

output :: ActivationFn -> [Double] -> Network -> [Double]
output f is (checkNet -> net) = output' is net
  where output' is' [] = is'
        output' is' (l:ls) = output' (map f $ netInputs is' l) ls

trainUntil :: Double -> Int -> ActivationFns -> Double -> TrainingSet -> Network -> Network
trainUntil maxErr lim gs r set = fst . trainUntil' maxErr lim gs r set

trainUntil' :: Double -> Int -> ActivationFns
    -> Double -> TrainingSet -> Network -> (Network, Double)
trainUntil' maxErr lim gs r set net
    | lim <= 1 || err <= maxErr = (net', err)
    | otherwise = trainUntil' maxErr (lim - 1) gs r set net
  where (net', err) = trainOnce' gs r set net

trainOnce :: ActivationFns -> Double -> TrainingSet -> Network -> Network
trainOnce gs r set = fst . trainOnce' gs r set

trainOnce' :: ActivationFns -> Double -> TrainingSet -> Network -> (Network, Double)
trainOnce' _ _ [] = error "Empty training set"
trainOnce' gs r [sam] = train gs r sam
trainOnce' gs r (sam:set) = train gs r sam . fst . trainOnce' gs r set

train :: ActivationFns -> Double -> TrainingSample -> Network -> (Network, Double)
train (g, g') r (is, ts) (checkNet -> net) = (snd $ bProp is net oss, err)
  where err = sum [ (t - o)^2 / 2 | t <- ts | o <- fst $ last oss ]
        oss = fProp is net
        fProp _ [] = []
        fProp is' (l:ls) = (os, map g' nis) : fProp os ls
          where nis = netInputs is' l
                os = map g nis
        bProp is' [l] [(os, os')] = (ds, [update is' l ds])
          where ds = [ o' * (t - o) | o <- os | o' <- os' | t <- ts ]
        bProp is' (cl:nl:ls) ((os, os'):oss') = (cds, update is' cl cds : net')
          where (pds, net') = bProp os (nl:ls) oss'
                cds = [ o' * sum [ ws !! i * pd | (ws, _) <- nl | pd <- pds ]
                    | o' <- os' | i <- [0..] ]
        bProp _ _ _ = undefined
        update is' l ds = [ (zipWith wUpdate ws is', wUpdate b 1)
            | (ws, b) <- l | d <- ds, let wUpdate w i = w + r * i * d ]
