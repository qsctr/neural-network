{-# LANGUAGE ViewPatterns #-}
{-# OPTIONS_GHC -Wno-type-defaults #-}

module AI.NeuralNetwork.ActivationFunctions
    ( ActivationFn, ActivationFn', ActivationFns
    , sigmoid, sigmoid'
    , tanh, tanh'
    ) where

type ActivationFn = Double -> Double
type ActivationFn' = Double -> Double
type ActivationFns = (ActivationFn, ActivationFn')

sigmoid :: ActivationFn
sigmoid x = 1 / (1 + exp (-x))

sigmoid' :: ActivationFn'
sigmoid' (sigmoid -> x) = x * (1 - x)

-- tanh is already in Prelude

tanh' :: ActivationFn'
tanh' x = 1 - (tanh x) ^ 2
