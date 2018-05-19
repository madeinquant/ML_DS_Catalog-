# -*- coding: utf-8 -*-
"""
Created on Fri May 18 01:07:39 2018

@author: santanu
"""
import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Input, LSTM, Dense
import numpy as np
import codecs



################# IN PROGRESS #############################################


def process_input_label(path):
    input_file = pd.read_json(path)
    
class video_captioning_nmt:
    
    def __init__(self):
        
        self.sequence_len = 100
        self.image_feat_len = 4096
        self.num_decoder_words = 20000
        self.latent_dim = 4096
        return None
        
    
        

    def model_(self):
        # Encoder Input will have the one hot encoded representation of the source input words in each timestep   
        encoder_inp = Input(shape=(None,self.image_feat_len),name='encoder_inp')
      # An LSTM has both a hidden state and a cell state at each timestep. We are going to extract the final hidden 
      # state and the final cell state from the LSTM to provide context to the decoder.The cell state is generally not 
      # available as one of the LSTM    outputs unless specified through "return_state=True"
        encoder = LSTM(self.latent_dim, return_sequences=True,name='encoder')
        encoder_out,states_h = encoder(encoder_inp)
        attention = dot([states_h,decoder_out], axes=[2, 2])
        attention = Activation('softmax')(attention)
        context = dot([attention,states_h], axes=[2,1])
        decoder_combined_inp = concatenate([decoder_out,context])
        #The decoder input is the target input sequence expressed as a sequence of one hot encoded vectors   
        #decoder_inp = Input(shape=(None,self.num_decoder_words))
        # The decoder needs to emit an output target word at every timestep and hence the parameter return_sequences is set to 
        # True. The decoder is trying to predict its own sequence but at a one time step gap. Also to provide context from the 
        # source language for translation the initial hidden and cell state of the decoder has been set with the last 
        # hidden and cell state of the encoder(that processed the source language text) 
        decoder_lstm = LSTM(self.latent_dim,return_sequences=True, return_state=True)
        decoder_out, decoder_state, _ = decoder_lstm(decoder_combined_inp,
                                             initial_state=[decoder_out,decoder_state])
        decoder_dense = Dense(self.num_decoder_words, activation='softmax')
        decoder_out = decoder_dense(decoder_out)
        model_enc_dec  = Model([encoder_inp, decoder_inp], decoder_out)
        encoder_model = Model(encoder_inp, encoder_states)
        return model_enc_dec,encoder_model,decoder_lstm,decoder_dense,decoder_inp
        
        
    def decoder_inference(self):
        decoder_state_input_h = Input(shape=(self.latent_dim,))
        decoder_state_input_c = Input(shape=(self.latent_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs,state_h,state_c = self.decoder_lstm(
        self.decoder_inp,initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = self.decoder_dense(decoder_outputs)
        decoder_model = Model(
        [self.decoder_inp] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)
        return decoder_model
        
        
    

# Reverse-lookup token index to decode sequences back to
# something readable.
        reverse_input_char_index = dict(
        (i, char) for char, i in input_token_index.items())
        reverse_target_char_index = dict(
        (i, char) for char, i in target_token_index.items())
  
        
        
        
        
        
    
    
    
    
    
    
    
    
    
    

    
    
    
    
