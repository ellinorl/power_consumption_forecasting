??$
??
:
Add
x"T
y"T
z"T"
Ttype:
2	
B
AddV2
x"T
y"T
z"T"
Ttype:
2	??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
:
Less
x"T
y"T
z
"
Ttype:
2	
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
:
Maximum
x"T
y"T
z"T"
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
:
Minimum
x"T
y"T
z"T"
Ttype:

2	
=
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type
output_handle"
element_dtypetype"

shape_typetype:
2	
?
TensorListReserve
element_shape"
shape_type
num_elements

handle"
element_dtypetype"

shape_typetype:
2	
?
TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint?????????
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?
?
While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
?"serve*2.4.12v2.4.0-49-g85c8b2a817f8??"
?
Dense_lay_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*#
shared_nameDense_lay_1/kernel
{
&Dense_lay_1/kernel/Read/ReadVariableOpReadVariableOpDense_lay_1/kernel* 
_output_shapes
:
??*
dtype0
y
Dense_lay_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*!
shared_nameDense_lay_1/bias
r
$Dense_lay_1/bias/Read/ReadVariableOpReadVariableOpDense_lay_1/bias*
_output_shapes	
:?*
dtype0

Output_lay/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?`*"
shared_nameOutput_lay/kernel
x
%Output_lay/kernel/Read/ReadVariableOpReadVariableOpOutput_lay/kernel*
_output_shapes
:	?`*
dtype0
v
Output_lay/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:`* 
shared_nameOutput_lay/bias
o
#Output_lay/bias/Read/ReadVariableOpReadVariableOpOutput_lay/bias*
_output_shapes
:`*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
?
LSTM_lay_0/lstm_cell/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*,
shared_nameLSTM_lay_0/lstm_cell/kernel
?
/LSTM_lay_0/lstm_cell/kernel/Read/ReadVariableOpReadVariableOpLSTM_lay_0/lstm_cell/kernel* 
_output_shapes
:
??*
dtype0
?
%LSTM_lay_0/lstm_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*6
shared_name'%LSTM_lay_0/lstm_cell/recurrent_kernel
?
9LSTM_lay_0/lstm_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOp%LSTM_lay_0/lstm_cell/recurrent_kernel* 
_output_shapes
:
??*
dtype0
?
LSTM_lay_0/lstm_cell/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?**
shared_nameLSTM_lay_0/lstm_cell/bias
?
-LSTM_lay_0/lstm_cell/bias/Read/ReadVariableOpReadVariableOpLSTM_lay_0/lstm_cell/bias*
_output_shapes	
:?*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
b
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0
?
Adam/Dense_lay_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??**
shared_nameAdam/Dense_lay_1/kernel/m
?
-Adam/Dense_lay_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Dense_lay_1/kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/Dense_lay_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*(
shared_nameAdam/Dense_lay_1/bias/m
?
+Adam/Dense_lay_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/Dense_lay_1/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/Output_lay/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?`*)
shared_nameAdam/Output_lay/kernel/m
?
,Adam/Output_lay/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Output_lay/kernel/m*
_output_shapes
:	?`*
dtype0
?
Adam/Output_lay/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*'
shared_nameAdam/Output_lay/bias/m
}
*Adam/Output_lay/bias/m/Read/ReadVariableOpReadVariableOpAdam/Output_lay/bias/m*
_output_shapes
:`*
dtype0
?
"Adam/LSTM_lay_0/lstm_cell/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*3
shared_name$"Adam/LSTM_lay_0/lstm_cell/kernel/m
?
6Adam/LSTM_lay_0/lstm_cell/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/LSTM_lay_0/lstm_cell/kernel/m* 
_output_shapes
:
??*
dtype0
?
,Adam/LSTM_lay_0/lstm_cell/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*=
shared_name.,Adam/LSTM_lay_0/lstm_cell/recurrent_kernel/m
?
@Adam/LSTM_lay_0/lstm_cell/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp,Adam/LSTM_lay_0/lstm_cell/recurrent_kernel/m* 
_output_shapes
:
??*
dtype0
?
 Adam/LSTM_lay_0/lstm_cell/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*1
shared_name" Adam/LSTM_lay_0/lstm_cell/bias/m
?
4Adam/LSTM_lay_0/lstm_cell/bias/m/Read/ReadVariableOpReadVariableOp Adam/LSTM_lay_0/lstm_cell/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/Dense_lay_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??**
shared_nameAdam/Dense_lay_1/kernel/v
?
-Adam/Dense_lay_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Dense_lay_1/kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/Dense_lay_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*(
shared_nameAdam/Dense_lay_1/bias/v
?
+Adam/Dense_lay_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/Dense_lay_1/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/Output_lay/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?`*)
shared_nameAdam/Output_lay/kernel/v
?
,Adam/Output_lay/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Output_lay/kernel/v*
_output_shapes
:	?`*
dtype0
?
Adam/Output_lay/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*'
shared_nameAdam/Output_lay/bias/v
}
*Adam/Output_lay/bias/v/Read/ReadVariableOpReadVariableOpAdam/Output_lay/bias/v*
_output_shapes
:`*
dtype0
?
"Adam/LSTM_lay_0/lstm_cell/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*3
shared_name$"Adam/LSTM_lay_0/lstm_cell/kernel/v
?
6Adam/LSTM_lay_0/lstm_cell/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/LSTM_lay_0/lstm_cell/kernel/v* 
_output_shapes
:
??*
dtype0
?
,Adam/LSTM_lay_0/lstm_cell/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*=
shared_name.,Adam/LSTM_lay_0/lstm_cell/recurrent_kernel/v
?
@Adam/LSTM_lay_0/lstm_cell/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp,Adam/LSTM_lay_0/lstm_cell/recurrent_kernel/v* 
_output_shapes
:
??*
dtype0
?
 Adam/LSTM_lay_0/lstm_cell/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*1
shared_name" Adam/LSTM_lay_0/lstm_cell/bias/v
?
4Adam/LSTM_lay_0/lstm_cell/bias/v/Read/ReadVariableOpReadVariableOp Adam/LSTM_lay_0/lstm_cell/bias/v*
_output_shapes	
:?*
dtype0

NoOpNoOp
?0
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?0
value?0B?0 B?0
?
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
	optimizer
trainable_variables
	variables
regularization_losses
		keras_api


signatures
l
cell

state_spec
trainable_variables
	variables
regularization_losses
	keras_api
R
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
trainable_variables
	variables
regularization_losses
 	keras_api
?
!iter

"beta_1

#beta_2
	$decay
%learning_ratem]m^m_m`&ma'mb(mcvdvevfvg&vh'vi(vj
1
&0
'1
(2
3
4
5
6
1
&0
'1
(2
3
4
5
6
 
?
trainable_variables
)layer_regularization_losses
*metrics
+non_trainable_variables

,layers
	variables
-layer_metrics
regularization_losses
 
~

&kernel
'recurrent_kernel
(bias
.trainable_variables
/	variables
0regularization_losses
1	keras_api
 

&0
'1
(2

&0
'1
(2
 
?
trainable_variables
2layer_regularization_losses
3metrics
4non_trainable_variables

5layers
	variables
6layer_metrics
regularization_losses

7states
 
 
 
?
trainable_variables
	variables
8layer_regularization_losses
9metrics

:layers
;non_trainable_variables
<layer_metrics
regularization_losses
^\
VARIABLE_VALUEDense_lay_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEDense_lay_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
trainable_variables
	variables
=layer_regularization_losses
>metrics

?layers
@non_trainable_variables
Alayer_metrics
regularization_losses
][
VARIABLE_VALUEOutput_lay/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEOutput_lay/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
trainable_variables
	variables
Blayer_regularization_losses
Cmetrics

Dlayers
Enon_trainable_variables
Flayer_metrics
regularization_losses
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUELSTM_lay_0/lstm_cell/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE%LSTM_lay_0/lstm_cell/recurrent_kernel0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUELSTM_lay_0/lstm_cell/bias0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
 

G0
H1
I2
 

0
1
2
3
 

&0
'1
(2

&0
'1
(2
 
?
.trainable_variables
/	variables
Jlayer_regularization_losses
Kmetrics

Llayers
Mnon_trainable_variables
Nlayer_metrics
0regularization_losses
 
 
 

0
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	Ototal
	Pcount
Q	variables
R	keras_api
D
	Stotal
	Tcount
U
_fn_kwargs
V	variables
W	keras_api
D
	Xtotal
	Ycount
Z
_fn_kwargs
[	variables
\	keras_api
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

O0
P1

Q	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

S0
T1

V	variables
QO
VARIABLE_VALUEtotal_24keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_24keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE
 

X0
Y1

[	variables
?
VARIABLE_VALUEAdam/Dense_lay_1/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/Dense_lay_1/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/Output_lay/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/Output_lay/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/LSTM_lay_0/lstm_cell/kernel/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE,Adam/LSTM_lay_0/lstm_cell/recurrent_kernel/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Adam/LSTM_lay_0/lstm_cell/bias/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdam/Dense_lay_1/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/Dense_lay_1/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/Output_lay/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/Output_lay/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/LSTM_lay_0/lstm_cell/kernel/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE,Adam/LSTM_lay_0/lstm_cell/recurrent_kernel/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Adam/LSTM_lay_0/lstm_cell/bias/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
 serving_default_LSTM_lay_0_inputPlaceholder*,
_output_shapes
:??????????*
dtype0*!
shape:??????????
?
StatefulPartitionedCallStatefulPartitionedCall serving_default_LSTM_lay_0_inputLSTM_lay_0/lstm_cell/kernelLSTM_lay_0/lstm_cell/bias%LSTM_lay_0/lstm_cell/recurrent_kernelDense_lay_1/kernelDense_lay_1/biasOutput_lay/kernelOutput_lay/bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????`*)
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8? *-
f(R&
$__inference_signature_wrapper_120189
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename&Dense_lay_1/kernel/Read/ReadVariableOp$Dense_lay_1/bias/Read/ReadVariableOp%Output_lay/kernel/Read/ReadVariableOp#Output_lay/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp/LSTM_lay_0/lstm_cell/kernel/Read/ReadVariableOp9LSTM_lay_0/lstm_cell/recurrent_kernel/Read/ReadVariableOp-LSTM_lay_0/lstm_cell/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOp-Adam/Dense_lay_1/kernel/m/Read/ReadVariableOp+Adam/Dense_lay_1/bias/m/Read/ReadVariableOp,Adam/Output_lay/kernel/m/Read/ReadVariableOp*Adam/Output_lay/bias/m/Read/ReadVariableOp6Adam/LSTM_lay_0/lstm_cell/kernel/m/Read/ReadVariableOp@Adam/LSTM_lay_0/lstm_cell/recurrent_kernel/m/Read/ReadVariableOp4Adam/LSTM_lay_0/lstm_cell/bias/m/Read/ReadVariableOp-Adam/Dense_lay_1/kernel/v/Read/ReadVariableOp+Adam/Dense_lay_1/bias/v/Read/ReadVariableOp,Adam/Output_lay/kernel/v/Read/ReadVariableOp*Adam/Output_lay/bias/v/Read/ReadVariableOp6Adam/LSTM_lay_0/lstm_cell/kernel/v/Read/ReadVariableOp@Adam/LSTM_lay_0/lstm_cell/recurrent_kernel/v/Read/ReadVariableOp4Adam/LSTM_lay_0/lstm_cell/bias/v/Read/ReadVariableOpConst*-
Tin&
$2"	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *(
f#R!
__inference__traced_save_122315
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameDense_lay_1/kernelDense_lay_1/biasOutput_lay/kernelOutput_lay/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rateLSTM_lay_0/lstm_cell/kernel%LSTM_lay_0/lstm_cell/recurrent_kernelLSTM_lay_0/lstm_cell/biastotalcounttotal_1count_1total_2count_2Adam/Dense_lay_1/kernel/mAdam/Dense_lay_1/bias/mAdam/Output_lay/kernel/mAdam/Output_lay/bias/m"Adam/LSTM_lay_0/lstm_cell/kernel/m,Adam/LSTM_lay_0/lstm_cell/recurrent_kernel/m Adam/LSTM_lay_0/lstm_cell/bias/mAdam/Dense_lay_1/kernel/vAdam/Dense_lay_1/bias/vAdam/Output_lay/kernel/vAdam/Output_lay/bias/v"Adam/LSTM_lay_0/lstm_cell/kernel/v,Adam/LSTM_lay_0/lstm_cell/recurrent_kernel/v Adam/LSTM_lay_0/lstm_cell/bias/v*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference__traced_restore_122421??!
?
?
+__inference_sequential_layer_call_fn_120779

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????`*)
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_1201022
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????`2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:??????????:::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
'sequential_LSTM_lay_0_while_body_118524H
Dsequential_lstm_lay_0_while_sequential_lstm_lay_0_while_loop_counterN
Jsequential_lstm_lay_0_while_sequential_lstm_lay_0_while_maximum_iterations+
'sequential_lstm_lay_0_while_placeholder-
)sequential_lstm_lay_0_while_placeholder_1-
)sequential_lstm_lay_0_while_placeholder_2-
)sequential_lstm_lay_0_while_placeholder_3G
Csequential_lstm_lay_0_while_sequential_lstm_lay_0_strided_slice_1_0?
sequential_lstm_lay_0_while_tensorarrayv2read_tensorlistgetitem_sequential_lstm_lay_0_tensorarrayunstack_tensorlistfromtensor_0I
Esequential_lstm_lay_0_while_lstm_cell_split_readvariableop_resource_0K
Gsequential_lstm_lay_0_while_lstm_cell_split_1_readvariableop_resource_0C
?sequential_lstm_lay_0_while_lstm_cell_readvariableop_resource_0(
$sequential_lstm_lay_0_while_identity*
&sequential_lstm_lay_0_while_identity_1*
&sequential_lstm_lay_0_while_identity_2*
&sequential_lstm_lay_0_while_identity_3*
&sequential_lstm_lay_0_while_identity_4*
&sequential_lstm_lay_0_while_identity_5E
Asequential_lstm_lay_0_while_sequential_lstm_lay_0_strided_slice_1?
}sequential_lstm_lay_0_while_tensorarrayv2read_tensorlistgetitem_sequential_lstm_lay_0_tensorarrayunstack_tensorlistfromtensorG
Csequential_lstm_lay_0_while_lstm_cell_split_readvariableop_resourceI
Esequential_lstm_lay_0_while_lstm_cell_split_1_readvariableop_resourceA
=sequential_lstm_lay_0_while_lstm_cell_readvariableop_resource??4sequential/LSTM_lay_0/while/lstm_cell/ReadVariableOp?6sequential/LSTM_lay_0/while/lstm_cell/ReadVariableOp_1?6sequential/LSTM_lay_0/while/lstm_cell/ReadVariableOp_2?6sequential/LSTM_lay_0/while/lstm_cell/ReadVariableOp_3?:sequential/LSTM_lay_0/while/lstm_cell/split/ReadVariableOp?<sequential/LSTM_lay_0/while/lstm_cell/split_1/ReadVariableOp?
Msequential/LSTM_lay_0/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2O
Msequential/LSTM_lay_0/while/TensorArrayV2Read/TensorListGetItem/element_shape?
?sequential/LSTM_lay_0/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsequential_lstm_lay_0_while_tensorarrayv2read_tensorlistgetitem_sequential_lstm_lay_0_tensorarrayunstack_tensorlistfromtensor_0'sequential_lstm_lay_0_while_placeholderVsequential/LSTM_lay_0/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02A
?sequential/LSTM_lay_0/while/TensorArrayV2Read/TensorListGetItem?
+sequential/LSTM_lay_0/while/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2-
+sequential/LSTM_lay_0/while/lstm_cell/Const?
5sequential/LSTM_lay_0/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :27
5sequential/LSTM_lay_0/while/lstm_cell/split/split_dim?
:sequential/LSTM_lay_0/while/lstm_cell/split/ReadVariableOpReadVariableOpEsequential_lstm_lay_0_while_lstm_cell_split_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02<
:sequential/LSTM_lay_0/while/lstm_cell/split/ReadVariableOp?
+sequential/LSTM_lay_0/while/lstm_cell/splitSplit>sequential/LSTM_lay_0/while/lstm_cell/split/split_dim:output:0Bsequential/LSTM_lay_0/while/lstm_cell/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
??:
??:
??:
??*
	num_split2-
+sequential/LSTM_lay_0/while/lstm_cell/split?
,sequential/LSTM_lay_0/while/lstm_cell/MatMulMatMulFsequential/LSTM_lay_0/while/TensorArrayV2Read/TensorListGetItem:item:04sequential/LSTM_lay_0/while/lstm_cell/split:output:0*
T0*(
_output_shapes
:??????????2.
,sequential/LSTM_lay_0/while/lstm_cell/MatMul?
.sequential/LSTM_lay_0/while/lstm_cell/MatMul_1MatMulFsequential/LSTM_lay_0/while/TensorArrayV2Read/TensorListGetItem:item:04sequential/LSTM_lay_0/while/lstm_cell/split:output:1*
T0*(
_output_shapes
:??????????20
.sequential/LSTM_lay_0/while/lstm_cell/MatMul_1?
.sequential/LSTM_lay_0/while/lstm_cell/MatMul_2MatMulFsequential/LSTM_lay_0/while/TensorArrayV2Read/TensorListGetItem:item:04sequential/LSTM_lay_0/while/lstm_cell/split:output:2*
T0*(
_output_shapes
:??????????20
.sequential/LSTM_lay_0/while/lstm_cell/MatMul_2?
.sequential/LSTM_lay_0/while/lstm_cell/MatMul_3MatMulFsequential/LSTM_lay_0/while/TensorArrayV2Read/TensorListGetItem:item:04sequential/LSTM_lay_0/while/lstm_cell/split:output:3*
T0*(
_output_shapes
:??????????20
.sequential/LSTM_lay_0/while/lstm_cell/MatMul_3?
-sequential/LSTM_lay_0/while/lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2/
-sequential/LSTM_lay_0/while/lstm_cell/Const_1?
7sequential/LSTM_lay_0/while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 29
7sequential/LSTM_lay_0/while/lstm_cell/split_1/split_dim?
<sequential/LSTM_lay_0/while/lstm_cell/split_1/ReadVariableOpReadVariableOpGsequential_lstm_lay_0_while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
:?*
dtype02>
<sequential/LSTM_lay_0/while/lstm_cell/split_1/ReadVariableOp?
-sequential/LSTM_lay_0/while/lstm_cell/split_1Split@sequential/LSTM_lay_0/while/lstm_cell/split_1/split_dim:output:0Dsequential/LSTM_lay_0/while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2/
-sequential/LSTM_lay_0/while/lstm_cell/split_1?
-sequential/LSTM_lay_0/while/lstm_cell/BiasAddBiasAdd6sequential/LSTM_lay_0/while/lstm_cell/MatMul:product:06sequential/LSTM_lay_0/while/lstm_cell/split_1:output:0*
T0*(
_output_shapes
:??????????2/
-sequential/LSTM_lay_0/while/lstm_cell/BiasAdd?
/sequential/LSTM_lay_0/while/lstm_cell/BiasAdd_1BiasAdd8sequential/LSTM_lay_0/while/lstm_cell/MatMul_1:product:06sequential/LSTM_lay_0/while/lstm_cell/split_1:output:1*
T0*(
_output_shapes
:??????????21
/sequential/LSTM_lay_0/while/lstm_cell/BiasAdd_1?
/sequential/LSTM_lay_0/while/lstm_cell/BiasAdd_2BiasAdd8sequential/LSTM_lay_0/while/lstm_cell/MatMul_2:product:06sequential/LSTM_lay_0/while/lstm_cell/split_1:output:2*
T0*(
_output_shapes
:??????????21
/sequential/LSTM_lay_0/while/lstm_cell/BiasAdd_2?
/sequential/LSTM_lay_0/while/lstm_cell/BiasAdd_3BiasAdd8sequential/LSTM_lay_0/while/lstm_cell/MatMul_3:product:06sequential/LSTM_lay_0/while/lstm_cell/split_1:output:3*
T0*(
_output_shapes
:??????????21
/sequential/LSTM_lay_0/while/lstm_cell/BiasAdd_3?
4sequential/LSTM_lay_0/while/lstm_cell/ReadVariableOpReadVariableOp?sequential_lstm_lay_0_while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype026
4sequential/LSTM_lay_0/while/lstm_cell/ReadVariableOp?
9sequential/LSTM_lay_0/while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2;
9sequential/LSTM_lay_0/while/lstm_cell/strided_slice/stack?
;sequential/LSTM_lay_0/while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2=
;sequential/LSTM_lay_0/while/lstm_cell/strided_slice/stack_1?
;sequential/LSTM_lay_0/while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2=
;sequential/LSTM_lay_0/while/lstm_cell/strided_slice/stack_2?
3sequential/LSTM_lay_0/while/lstm_cell/strided_sliceStridedSlice<sequential/LSTM_lay_0/while/lstm_cell/ReadVariableOp:value:0Bsequential/LSTM_lay_0/while/lstm_cell/strided_slice/stack:output:0Dsequential/LSTM_lay_0/while/lstm_cell/strided_slice/stack_1:output:0Dsequential/LSTM_lay_0/while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask25
3sequential/LSTM_lay_0/while/lstm_cell/strided_slice?
.sequential/LSTM_lay_0/while/lstm_cell/MatMul_4MatMul)sequential_lstm_lay_0_while_placeholder_2<sequential/LSTM_lay_0/while/lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:??????????20
.sequential/LSTM_lay_0/while/lstm_cell/MatMul_4?
)sequential/LSTM_lay_0/while/lstm_cell/addAddV26sequential/LSTM_lay_0/while/lstm_cell/BiasAdd:output:08sequential/LSTM_lay_0/while/lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2+
)sequential/LSTM_lay_0/while/lstm_cell/add?
-sequential/LSTM_lay_0/while/lstm_cell/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2/
-sequential/LSTM_lay_0/while/lstm_cell/Const_2?
-sequential/LSTM_lay_0/while/lstm_cell/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2/
-sequential/LSTM_lay_0/while/lstm_cell/Const_3?
)sequential/LSTM_lay_0/while/lstm_cell/MulMul-sequential/LSTM_lay_0/while/lstm_cell/add:z:06sequential/LSTM_lay_0/while/lstm_cell/Const_2:output:0*
T0*(
_output_shapes
:??????????2+
)sequential/LSTM_lay_0/while/lstm_cell/Mul?
+sequential/LSTM_lay_0/while/lstm_cell/Add_1Add-sequential/LSTM_lay_0/while/lstm_cell/Mul:z:06sequential/LSTM_lay_0/while/lstm_cell/Const_3:output:0*
T0*(
_output_shapes
:??????????2-
+sequential/LSTM_lay_0/while/lstm_cell/Add_1?
=sequential/LSTM_lay_0/while/lstm_cell/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2?
=sequential/LSTM_lay_0/while/lstm_cell/clip_by_value/Minimum/y?
;sequential/LSTM_lay_0/while/lstm_cell/clip_by_value/MinimumMinimum/sequential/LSTM_lay_0/while/lstm_cell/Add_1:z:0Fsequential/LSTM_lay_0/while/lstm_cell/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:??????????2=
;sequential/LSTM_lay_0/while/lstm_cell/clip_by_value/Minimum?
5sequential/LSTM_lay_0/while/lstm_cell/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    27
5sequential/LSTM_lay_0/while/lstm_cell/clip_by_value/y?
3sequential/LSTM_lay_0/while/lstm_cell/clip_by_valueMaximum?sequential/LSTM_lay_0/while/lstm_cell/clip_by_value/Minimum:z:0>sequential/LSTM_lay_0/while/lstm_cell/clip_by_value/y:output:0*
T0*(
_output_shapes
:??????????25
3sequential/LSTM_lay_0/while/lstm_cell/clip_by_value?
6sequential/LSTM_lay_0/while/lstm_cell/ReadVariableOp_1ReadVariableOp?sequential_lstm_lay_0_while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype028
6sequential/LSTM_lay_0/while/lstm_cell/ReadVariableOp_1?
;sequential/LSTM_lay_0/while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2=
;sequential/LSTM_lay_0/while/lstm_cell/strided_slice_1/stack?
=sequential/LSTM_lay_0/while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2?
=sequential/LSTM_lay_0/while/lstm_cell/strided_slice_1/stack_1?
=sequential/LSTM_lay_0/while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2?
=sequential/LSTM_lay_0/while/lstm_cell/strided_slice_1/stack_2?
5sequential/LSTM_lay_0/while/lstm_cell/strided_slice_1StridedSlice>sequential/LSTM_lay_0/while/lstm_cell/ReadVariableOp_1:value:0Dsequential/LSTM_lay_0/while/lstm_cell/strided_slice_1/stack:output:0Fsequential/LSTM_lay_0/while/lstm_cell/strided_slice_1/stack_1:output:0Fsequential/LSTM_lay_0/while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask27
5sequential/LSTM_lay_0/while/lstm_cell/strided_slice_1?
.sequential/LSTM_lay_0/while/lstm_cell/MatMul_5MatMul)sequential_lstm_lay_0_while_placeholder_2>sequential/LSTM_lay_0/while/lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????20
.sequential/LSTM_lay_0/while/lstm_cell/MatMul_5?
+sequential/LSTM_lay_0/while/lstm_cell/add_2AddV28sequential/LSTM_lay_0/while/lstm_cell/BiasAdd_1:output:08sequential/LSTM_lay_0/while/lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2-
+sequential/LSTM_lay_0/while/lstm_cell/add_2?
-sequential/LSTM_lay_0/while/lstm_cell/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2/
-sequential/LSTM_lay_0/while/lstm_cell/Const_4?
-sequential/LSTM_lay_0/while/lstm_cell/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2/
-sequential/LSTM_lay_0/while/lstm_cell/Const_5?
+sequential/LSTM_lay_0/while/lstm_cell/Mul_1Mul/sequential/LSTM_lay_0/while/lstm_cell/add_2:z:06sequential/LSTM_lay_0/while/lstm_cell/Const_4:output:0*
T0*(
_output_shapes
:??????????2-
+sequential/LSTM_lay_0/while/lstm_cell/Mul_1?
+sequential/LSTM_lay_0/while/lstm_cell/Add_3Add/sequential/LSTM_lay_0/while/lstm_cell/Mul_1:z:06sequential/LSTM_lay_0/while/lstm_cell/Const_5:output:0*
T0*(
_output_shapes
:??????????2-
+sequential/LSTM_lay_0/while/lstm_cell/Add_3?
?sequential/LSTM_lay_0/while/lstm_cell/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2A
?sequential/LSTM_lay_0/while/lstm_cell/clip_by_value_1/Minimum/y?
=sequential/LSTM_lay_0/while/lstm_cell/clip_by_value_1/MinimumMinimum/sequential/LSTM_lay_0/while/lstm_cell/Add_3:z:0Hsequential/LSTM_lay_0/while/lstm_cell/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:??????????2?
=sequential/LSTM_lay_0/while/lstm_cell/clip_by_value_1/Minimum?
7sequential/LSTM_lay_0/while/lstm_cell/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    29
7sequential/LSTM_lay_0/while/lstm_cell/clip_by_value_1/y?
5sequential/LSTM_lay_0/while/lstm_cell/clip_by_value_1MaximumAsequential/LSTM_lay_0/while/lstm_cell/clip_by_value_1/Minimum:z:0@sequential/LSTM_lay_0/while/lstm_cell/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:??????????27
5sequential/LSTM_lay_0/while/lstm_cell/clip_by_value_1?
+sequential/LSTM_lay_0/while/lstm_cell/mul_2Mul9sequential/LSTM_lay_0/while/lstm_cell/clip_by_value_1:z:0)sequential_lstm_lay_0_while_placeholder_3*
T0*(
_output_shapes
:??????????2-
+sequential/LSTM_lay_0/while/lstm_cell/mul_2?
6sequential/LSTM_lay_0/while/lstm_cell/ReadVariableOp_2ReadVariableOp?sequential_lstm_lay_0_while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype028
6sequential/LSTM_lay_0/while/lstm_cell/ReadVariableOp_2?
;sequential/LSTM_lay_0/while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2=
;sequential/LSTM_lay_0/while/lstm_cell/strided_slice_2/stack?
=sequential/LSTM_lay_0/while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2?
=sequential/LSTM_lay_0/while/lstm_cell/strided_slice_2/stack_1?
=sequential/LSTM_lay_0/while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2?
=sequential/LSTM_lay_0/while/lstm_cell/strided_slice_2/stack_2?
5sequential/LSTM_lay_0/while/lstm_cell/strided_slice_2StridedSlice>sequential/LSTM_lay_0/while/lstm_cell/ReadVariableOp_2:value:0Dsequential/LSTM_lay_0/while/lstm_cell/strided_slice_2/stack:output:0Fsequential/LSTM_lay_0/while/lstm_cell/strided_slice_2/stack_1:output:0Fsequential/LSTM_lay_0/while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask27
5sequential/LSTM_lay_0/while/lstm_cell/strided_slice_2?
.sequential/LSTM_lay_0/while/lstm_cell/MatMul_6MatMul)sequential_lstm_lay_0_while_placeholder_2>sequential/LSTM_lay_0/while/lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????20
.sequential/LSTM_lay_0/while/lstm_cell/MatMul_6?
+sequential/LSTM_lay_0/while/lstm_cell/add_4AddV28sequential/LSTM_lay_0/while/lstm_cell/BiasAdd_2:output:08sequential/LSTM_lay_0/while/lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2-
+sequential/LSTM_lay_0/while/lstm_cell/add_4?
-sequential/LSTM_lay_0/while/lstm_cell/SigmoidSigmoid/sequential/LSTM_lay_0/while/lstm_cell/add_4:z:0*
T0*(
_output_shapes
:??????????2/
-sequential/LSTM_lay_0/while/lstm_cell/Sigmoid?
+sequential/LSTM_lay_0/while/lstm_cell/mul_3Mul7sequential/LSTM_lay_0/while/lstm_cell/clip_by_value:z:01sequential/LSTM_lay_0/while/lstm_cell/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2-
+sequential/LSTM_lay_0/while/lstm_cell/mul_3?
+sequential/LSTM_lay_0/while/lstm_cell/add_5AddV2/sequential/LSTM_lay_0/while/lstm_cell/mul_2:z:0/sequential/LSTM_lay_0/while/lstm_cell/mul_3:z:0*
T0*(
_output_shapes
:??????????2-
+sequential/LSTM_lay_0/while/lstm_cell/add_5?
6sequential/LSTM_lay_0/while/lstm_cell/ReadVariableOp_3ReadVariableOp?sequential_lstm_lay_0_while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype028
6sequential/LSTM_lay_0/while/lstm_cell/ReadVariableOp_3?
;sequential/LSTM_lay_0/while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2=
;sequential/LSTM_lay_0/while/lstm_cell/strided_slice_3/stack?
=sequential/LSTM_lay_0/while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2?
=sequential/LSTM_lay_0/while/lstm_cell/strided_slice_3/stack_1?
=sequential/LSTM_lay_0/while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2?
=sequential/LSTM_lay_0/while/lstm_cell/strided_slice_3/stack_2?
5sequential/LSTM_lay_0/while/lstm_cell/strided_slice_3StridedSlice>sequential/LSTM_lay_0/while/lstm_cell/ReadVariableOp_3:value:0Dsequential/LSTM_lay_0/while/lstm_cell/strided_slice_3/stack:output:0Fsequential/LSTM_lay_0/while/lstm_cell/strided_slice_3/stack_1:output:0Fsequential/LSTM_lay_0/while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask27
5sequential/LSTM_lay_0/while/lstm_cell/strided_slice_3?
.sequential/LSTM_lay_0/while/lstm_cell/MatMul_7MatMul)sequential_lstm_lay_0_while_placeholder_2>sequential/LSTM_lay_0/while/lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????20
.sequential/LSTM_lay_0/while/lstm_cell/MatMul_7?
+sequential/LSTM_lay_0/while/lstm_cell/add_6AddV28sequential/LSTM_lay_0/while/lstm_cell/BiasAdd_3:output:08sequential/LSTM_lay_0/while/lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2-
+sequential/LSTM_lay_0/while/lstm_cell/add_6?
-sequential/LSTM_lay_0/while/lstm_cell/Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2/
-sequential/LSTM_lay_0/while/lstm_cell/Const_6?
-sequential/LSTM_lay_0/while/lstm_cell/Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2/
-sequential/LSTM_lay_0/while/lstm_cell/Const_7?
+sequential/LSTM_lay_0/while/lstm_cell/Mul_4Mul/sequential/LSTM_lay_0/while/lstm_cell/add_6:z:06sequential/LSTM_lay_0/while/lstm_cell/Const_6:output:0*
T0*(
_output_shapes
:??????????2-
+sequential/LSTM_lay_0/while/lstm_cell/Mul_4?
+sequential/LSTM_lay_0/while/lstm_cell/Add_7Add/sequential/LSTM_lay_0/while/lstm_cell/Mul_4:z:06sequential/LSTM_lay_0/while/lstm_cell/Const_7:output:0*
T0*(
_output_shapes
:??????????2-
+sequential/LSTM_lay_0/while/lstm_cell/Add_7?
?sequential/LSTM_lay_0/while/lstm_cell/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2A
?sequential/LSTM_lay_0/while/lstm_cell/clip_by_value_2/Minimum/y?
=sequential/LSTM_lay_0/while/lstm_cell/clip_by_value_2/MinimumMinimum/sequential/LSTM_lay_0/while/lstm_cell/Add_7:z:0Hsequential/LSTM_lay_0/while/lstm_cell/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:??????????2?
=sequential/LSTM_lay_0/while/lstm_cell/clip_by_value_2/Minimum?
7sequential/LSTM_lay_0/while/lstm_cell/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    29
7sequential/LSTM_lay_0/while/lstm_cell/clip_by_value_2/y?
5sequential/LSTM_lay_0/while/lstm_cell/clip_by_value_2MaximumAsequential/LSTM_lay_0/while/lstm_cell/clip_by_value_2/Minimum:z:0@sequential/LSTM_lay_0/while/lstm_cell/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:??????????27
5sequential/LSTM_lay_0/while/lstm_cell/clip_by_value_2?
/sequential/LSTM_lay_0/while/lstm_cell/Sigmoid_1Sigmoid/sequential/LSTM_lay_0/while/lstm_cell/add_5:z:0*
T0*(
_output_shapes
:??????????21
/sequential/LSTM_lay_0/while/lstm_cell/Sigmoid_1?
+sequential/LSTM_lay_0/while/lstm_cell/mul_5Mul9sequential/LSTM_lay_0/while/lstm_cell/clip_by_value_2:z:03sequential/LSTM_lay_0/while/lstm_cell/Sigmoid_1:y:0*
T0*(
_output_shapes
:??????????2-
+sequential/LSTM_lay_0/while/lstm_cell/mul_5?
@sequential/LSTM_lay_0/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)sequential_lstm_lay_0_while_placeholder_1'sequential_lstm_lay_0_while_placeholder/sequential/LSTM_lay_0/while/lstm_cell/mul_5:z:0*
_output_shapes
: *
element_dtype02B
@sequential/LSTM_lay_0/while/TensorArrayV2Write/TensorListSetItem?
!sequential/LSTM_lay_0/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2#
!sequential/LSTM_lay_0/while/add/y?
sequential/LSTM_lay_0/while/addAddV2'sequential_lstm_lay_0_while_placeholder*sequential/LSTM_lay_0/while/add/y:output:0*
T0*
_output_shapes
: 2!
sequential/LSTM_lay_0/while/add?
#sequential/LSTM_lay_0/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2%
#sequential/LSTM_lay_0/while/add_1/y?
!sequential/LSTM_lay_0/while/add_1AddV2Dsequential_lstm_lay_0_while_sequential_lstm_lay_0_while_loop_counter,sequential/LSTM_lay_0/while/add_1/y:output:0*
T0*
_output_shapes
: 2#
!sequential/LSTM_lay_0/while/add_1?
$sequential/LSTM_lay_0/while/IdentityIdentity%sequential/LSTM_lay_0/while/add_1:z:05^sequential/LSTM_lay_0/while/lstm_cell/ReadVariableOp7^sequential/LSTM_lay_0/while/lstm_cell/ReadVariableOp_17^sequential/LSTM_lay_0/while/lstm_cell/ReadVariableOp_27^sequential/LSTM_lay_0/while/lstm_cell/ReadVariableOp_3;^sequential/LSTM_lay_0/while/lstm_cell/split/ReadVariableOp=^sequential/LSTM_lay_0/while/lstm_cell/split_1/ReadVariableOp*
T0*
_output_shapes
: 2&
$sequential/LSTM_lay_0/while/Identity?
&sequential/LSTM_lay_0/while/Identity_1IdentityJsequential_lstm_lay_0_while_sequential_lstm_lay_0_while_maximum_iterations5^sequential/LSTM_lay_0/while/lstm_cell/ReadVariableOp7^sequential/LSTM_lay_0/while/lstm_cell/ReadVariableOp_17^sequential/LSTM_lay_0/while/lstm_cell/ReadVariableOp_27^sequential/LSTM_lay_0/while/lstm_cell/ReadVariableOp_3;^sequential/LSTM_lay_0/while/lstm_cell/split/ReadVariableOp=^sequential/LSTM_lay_0/while/lstm_cell/split_1/ReadVariableOp*
T0*
_output_shapes
: 2(
&sequential/LSTM_lay_0/while/Identity_1?
&sequential/LSTM_lay_0/while/Identity_2Identity#sequential/LSTM_lay_0/while/add:z:05^sequential/LSTM_lay_0/while/lstm_cell/ReadVariableOp7^sequential/LSTM_lay_0/while/lstm_cell/ReadVariableOp_17^sequential/LSTM_lay_0/while/lstm_cell/ReadVariableOp_27^sequential/LSTM_lay_0/while/lstm_cell/ReadVariableOp_3;^sequential/LSTM_lay_0/while/lstm_cell/split/ReadVariableOp=^sequential/LSTM_lay_0/while/lstm_cell/split_1/ReadVariableOp*
T0*
_output_shapes
: 2(
&sequential/LSTM_lay_0/while/Identity_2?
&sequential/LSTM_lay_0/while/Identity_3IdentityPsequential/LSTM_lay_0/while/TensorArrayV2Write/TensorListSetItem:output_handle:05^sequential/LSTM_lay_0/while/lstm_cell/ReadVariableOp7^sequential/LSTM_lay_0/while/lstm_cell/ReadVariableOp_17^sequential/LSTM_lay_0/while/lstm_cell/ReadVariableOp_27^sequential/LSTM_lay_0/while/lstm_cell/ReadVariableOp_3;^sequential/LSTM_lay_0/while/lstm_cell/split/ReadVariableOp=^sequential/LSTM_lay_0/while/lstm_cell/split_1/ReadVariableOp*
T0*
_output_shapes
: 2(
&sequential/LSTM_lay_0/while/Identity_3?
&sequential/LSTM_lay_0/while/Identity_4Identity/sequential/LSTM_lay_0/while/lstm_cell/mul_5:z:05^sequential/LSTM_lay_0/while/lstm_cell/ReadVariableOp7^sequential/LSTM_lay_0/while/lstm_cell/ReadVariableOp_17^sequential/LSTM_lay_0/while/lstm_cell/ReadVariableOp_27^sequential/LSTM_lay_0/while/lstm_cell/ReadVariableOp_3;^sequential/LSTM_lay_0/while/lstm_cell/split/ReadVariableOp=^sequential/LSTM_lay_0/while/lstm_cell/split_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2(
&sequential/LSTM_lay_0/while/Identity_4?
&sequential/LSTM_lay_0/while/Identity_5Identity/sequential/LSTM_lay_0/while/lstm_cell/add_5:z:05^sequential/LSTM_lay_0/while/lstm_cell/ReadVariableOp7^sequential/LSTM_lay_0/while/lstm_cell/ReadVariableOp_17^sequential/LSTM_lay_0/while/lstm_cell/ReadVariableOp_27^sequential/LSTM_lay_0/while/lstm_cell/ReadVariableOp_3;^sequential/LSTM_lay_0/while/lstm_cell/split/ReadVariableOp=^sequential/LSTM_lay_0/while/lstm_cell/split_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2(
&sequential/LSTM_lay_0/while/Identity_5"U
$sequential_lstm_lay_0_while_identity-sequential/LSTM_lay_0/while/Identity:output:0"Y
&sequential_lstm_lay_0_while_identity_1/sequential/LSTM_lay_0/while/Identity_1:output:0"Y
&sequential_lstm_lay_0_while_identity_2/sequential/LSTM_lay_0/while/Identity_2:output:0"Y
&sequential_lstm_lay_0_while_identity_3/sequential/LSTM_lay_0/while/Identity_3:output:0"Y
&sequential_lstm_lay_0_while_identity_4/sequential/LSTM_lay_0/while/Identity_4:output:0"Y
&sequential_lstm_lay_0_while_identity_5/sequential/LSTM_lay_0/while/Identity_5:output:0"?
=sequential_lstm_lay_0_while_lstm_cell_readvariableop_resource?sequential_lstm_lay_0_while_lstm_cell_readvariableop_resource_0"?
Esequential_lstm_lay_0_while_lstm_cell_split_1_readvariableop_resourceGsequential_lstm_lay_0_while_lstm_cell_split_1_readvariableop_resource_0"?
Csequential_lstm_lay_0_while_lstm_cell_split_readvariableop_resourceEsequential_lstm_lay_0_while_lstm_cell_split_readvariableop_resource_0"?
Asequential_lstm_lay_0_while_sequential_lstm_lay_0_strided_slice_1Csequential_lstm_lay_0_while_sequential_lstm_lay_0_strided_slice_1_0"?
}sequential_lstm_lay_0_while_tensorarrayv2read_tensorlistgetitem_sequential_lstm_lay_0_tensorarrayunstack_tensorlistfromtensorsequential_lstm_lay_0_while_tensorarrayv2read_tensorlistgetitem_sequential_lstm_lay_0_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :??????????:??????????: : :::2l
4sequential/LSTM_lay_0/while/lstm_cell/ReadVariableOp4sequential/LSTM_lay_0/while/lstm_cell/ReadVariableOp2p
6sequential/LSTM_lay_0/while/lstm_cell/ReadVariableOp_16sequential/LSTM_lay_0/while/lstm_cell/ReadVariableOp_12p
6sequential/LSTM_lay_0/while/lstm_cell/ReadVariableOp_26sequential/LSTM_lay_0/while/lstm_cell/ReadVariableOp_22p
6sequential/LSTM_lay_0/while/lstm_cell/ReadVariableOp_36sequential/LSTM_lay_0/while/lstm_cell/ReadVariableOp_32x
:sequential/LSTM_lay_0/while/lstm_cell/split/ReadVariableOp:sequential/LSTM_lay_0/while/lstm_cell/split/ReadVariableOp2|
<sequential/LSTM_lay_0/while/lstm_cell/split_1/ReadVariableOp<sequential/LSTM_lay_0/while/lstm_cell/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
+__inference_LSTM_lay_0_layer_call_fn_121903
inputs_0
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_LSTM_lay_0_layer_call_and_return_conditional_losses_1192622
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:???????????????????:::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:???????????????????
"
_user_specified_name
inputs/0
?	
?
G__inference_Dense_lay_1_layer_call_and_return_conditional_losses_120012

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
G__inference_Dense_lay_1_layer_call_and_return_conditional_losses_121952

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?G
?
__inference__traced_save_122315
file_prefix1
-savev2_dense_lay_1_kernel_read_readvariableop/
+savev2_dense_lay_1_bias_read_readvariableop0
,savev2_output_lay_kernel_read_readvariableop.
*savev2_output_lay_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop:
6savev2_lstm_lay_0_lstm_cell_kernel_read_readvariableopD
@savev2_lstm_lay_0_lstm_cell_recurrent_kernel_read_readvariableop8
4savev2_lstm_lay_0_lstm_cell_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop8
4savev2_adam_dense_lay_1_kernel_m_read_readvariableop6
2savev2_adam_dense_lay_1_bias_m_read_readvariableop7
3savev2_adam_output_lay_kernel_m_read_readvariableop5
1savev2_adam_output_lay_bias_m_read_readvariableopA
=savev2_adam_lstm_lay_0_lstm_cell_kernel_m_read_readvariableopK
Gsavev2_adam_lstm_lay_0_lstm_cell_recurrent_kernel_m_read_readvariableop?
;savev2_adam_lstm_lay_0_lstm_cell_bias_m_read_readvariableop8
4savev2_adam_dense_lay_1_kernel_v_read_readvariableop6
2savev2_adam_dense_lay_1_bias_v_read_readvariableop7
3savev2_adam_output_lay_kernel_v_read_readvariableop5
1savev2_adam_output_lay_bias_v_read_readvariableopA
=savev2_adam_lstm_lay_0_lstm_cell_kernel_v_read_readvariableopK
Gsavev2_adam_lstm_lay_0_lstm_cell_recurrent_kernel_v_read_readvariableop?
;savev2_adam_lstm_lay_0_lstm_cell_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:!*
dtype0*?
value?B?!B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:!*
dtype0*U
valueLBJ!B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0-savev2_dense_lay_1_kernel_read_readvariableop+savev2_dense_lay_1_bias_read_readvariableop,savev2_output_lay_kernel_read_readvariableop*savev2_output_lay_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop6savev2_lstm_lay_0_lstm_cell_kernel_read_readvariableop@savev2_lstm_lay_0_lstm_cell_recurrent_kernel_read_readvariableop4savev2_lstm_lay_0_lstm_cell_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop4savev2_adam_dense_lay_1_kernel_m_read_readvariableop2savev2_adam_dense_lay_1_bias_m_read_readvariableop3savev2_adam_output_lay_kernel_m_read_readvariableop1savev2_adam_output_lay_bias_m_read_readvariableop=savev2_adam_lstm_lay_0_lstm_cell_kernel_m_read_readvariableopGsavev2_adam_lstm_lay_0_lstm_cell_recurrent_kernel_m_read_readvariableop;savev2_adam_lstm_lay_0_lstm_cell_bias_m_read_readvariableop4savev2_adam_dense_lay_1_kernel_v_read_readvariableop2savev2_adam_dense_lay_1_bias_v_read_readvariableop3savev2_adam_output_lay_kernel_v_read_readvariableop1savev2_adam_output_lay_bias_v_read_readvariableop=savev2_adam_lstm_lay_0_lstm_cell_kernel_v_read_readvariableopGsavev2_adam_lstm_lay_0_lstm_cell_recurrent_kernel_v_read_readvariableop;savev2_adam_lstm_lay_0_lstm_cell_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 */
dtypes%
#2!	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :
??:?:	?`:`: : : : : :
??:
??:?: : : : : : :
??:?:	?`:`:
??:
??:?:
??:?:	?`:`:
??:
??:?: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?`: 

_output_shapes
:`:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :&
"
 
_output_shapes
:
??:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?`: 

_output_shapes
:`:&"
 
_output_shapes
:
??:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?`: 

_output_shapes
:`:&"
 
_output_shapes
:
??:&"
 
_output_shapes
:
??:! 

_output_shapes	
:?:!

_output_shapes
: 
?
?
'sequential_LSTM_lay_0_while_cond_118523H
Dsequential_lstm_lay_0_while_sequential_lstm_lay_0_while_loop_counterN
Jsequential_lstm_lay_0_while_sequential_lstm_lay_0_while_maximum_iterations+
'sequential_lstm_lay_0_while_placeholder-
)sequential_lstm_lay_0_while_placeholder_1-
)sequential_lstm_lay_0_while_placeholder_2-
)sequential_lstm_lay_0_while_placeholder_3J
Fsequential_lstm_lay_0_while_less_sequential_lstm_lay_0_strided_slice_1`
\sequential_lstm_lay_0_while_sequential_lstm_lay_0_while_cond_118523___redundant_placeholder0`
\sequential_lstm_lay_0_while_sequential_lstm_lay_0_while_cond_118523___redundant_placeholder1`
\sequential_lstm_lay_0_while_sequential_lstm_lay_0_while_cond_118523___redundant_placeholder2`
\sequential_lstm_lay_0_while_sequential_lstm_lay_0_while_cond_118523___redundant_placeholder3(
$sequential_lstm_lay_0_while_identity
?
 sequential/LSTM_lay_0/while/LessLess'sequential_lstm_lay_0_while_placeholderFsequential_lstm_lay_0_while_less_sequential_lstm_lay_0_strided_slice_1*
T0*
_output_shapes
: 2"
 sequential/LSTM_lay_0/while/Less?
$sequential/LSTM_lay_0/while/IdentityIdentity$sequential/LSTM_lay_0/while/Less:z:0*
T0
*
_output_shapes
: 2&
$sequential/LSTM_lay_0/while/Identity"U
$sequential_lstm_lay_0_while_identity-sequential/LSTM_lay_0/while/Identity:output:0*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
$__inference_signature_wrapper_120189
lstm_lay_0_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalllstm_lay_0_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????`*)
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__wrapped_model_1186802
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????`2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:??????????:::::::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
,
_output_shapes
:??????????
*
_user_specified_nameLSTM_lay_0_input
??
?
F__inference_LSTM_lay_0_layer_call_and_return_conditional_losses_119673

inputs+
'lstm_cell_split_readvariableop_resource-
)lstm_cell_split_1_readvariableop_resource%
!lstm_cell_readvariableop_resource
identity??lstm_cell/ReadVariableOp?lstm_cell/ReadVariableOp_1?lstm_cell/ReadVariableOp_2?lstm_cell/ReadVariableOp_3?lstm_cell/split/ReadVariableOp? lstm_cell/split_1/ReadVariableOp?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:??????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_2d
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dim?
lstm_cell/split/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
lstm_cell/split/ReadVariableOp?
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
??:
??:
??:
??*
	num_split2
lstm_cell/split?
lstm_cell/MatMulMatMulstrided_slice_2:output:0lstm_cell/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul?
lstm_cell/MatMul_1MatMulstrided_slice_2:output:0lstm_cell/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_1?
lstm_cell/MatMul_2MatMulstrided_slice_2:output:0lstm_cell/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_2?
lstm_cell/MatMul_3MatMulstrided_slice_2:output:0lstm_cell/split:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_3h
lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Const_1|
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell/split_1/split_dim?
 lstm_cell/split_1/ReadVariableOpReadVariableOp)lstm_cell_split_1_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 lstm_cell/split_1/ReadVariableOp?
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2
lstm_cell/split_1?
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd?
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_1?
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_2?
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_3?
lstm_cell/ReadVariableOpReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp?
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
lstm_cell/strided_slice/stack?
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2!
lstm_cell/strided_slice/stack_1?
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
lstm_cell/strided_slice/stack_2?
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice?
lstm_cell/MatMul_4MatMulzeros:output:0 lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_4?
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/addk
lstm_cell/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
lstm_cell/Const_2k
lstm_cell/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm_cell/Const_3?
lstm_cell/MulMullstm_cell/add:z:0lstm_cell/Const_2:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Mul?
lstm_cell/Add_1Addlstm_cell/Mul:z:0lstm_cell/Const_3:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Add_1?
!lstm_cell/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2#
!lstm_cell/clip_by_value/Minimum/y?
lstm_cell/clip_by_value/MinimumMinimumlstm_cell/Add_1:z:0*lstm_cell/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:??????????2!
lstm_cell/clip_by_value/Minimum{
lstm_cell/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_cell/clip_by_value/y?
lstm_cell/clip_by_valueMaximum#lstm_cell/clip_by_value/Minimum:z:0"lstm_cell/clip_by_value/y:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/clip_by_value?
lstm_cell/ReadVariableOp_1ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_1?
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2!
lstm_cell/strided_slice_1/stack?
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_1/stack_1?
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_1/stack_2?
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_1?
lstm_cell/MatMul_5MatMulzeros:output:0"lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_5?
lstm_cell/add_2AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_2k
lstm_cell/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
lstm_cell/Const_4k
lstm_cell/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm_cell/Const_5?
lstm_cell/Mul_1Mullstm_cell/add_2:z:0lstm_cell/Const_4:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Mul_1?
lstm_cell/Add_3Addlstm_cell/Mul_1:z:0lstm_cell/Const_5:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Add_3?
#lstm_cell/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2%
#lstm_cell/clip_by_value_1/Minimum/y?
!lstm_cell/clip_by_value_1/MinimumMinimumlstm_cell/Add_3:z:0,lstm_cell/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:??????????2#
!lstm_cell/clip_by_value_1/Minimum
lstm_cell/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_cell/clip_by_value_1/y?
lstm_cell/clip_by_value_1Maximum%lstm_cell/clip_by_value_1/Minimum:z:0$lstm_cell/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/clip_by_value_1?
lstm_cell/mul_2Mullstm_cell/clip_by_value_1:z:0zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_2?
lstm_cell/ReadVariableOp_2ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_2?
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_2/stack?
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2#
!lstm_cell/strided_slice_2/stack_1?
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_2/stack_2?
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_2?
lstm_cell/MatMul_6MatMulzeros:output:0"lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_6?
lstm_cell/add_4AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_4y
lstm_cell/SigmoidSigmoidlstm_cell/add_4:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid?
lstm_cell/mul_3Mullstm_cell/clip_by_value:z:0lstm_cell/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_3?
lstm_cell/add_5AddV2lstm_cell/mul_2:z:0lstm_cell/mul_3:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_5?
lstm_cell/ReadVariableOp_3ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_3?
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2!
lstm_cell/strided_slice_3/stack?
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_3/stack_1?
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_3/stack_2?
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_3?
lstm_cell/MatMul_7MatMulzeros:output:0"lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_7?
lstm_cell/add_6AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_6k
lstm_cell/Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
lstm_cell/Const_6k
lstm_cell/Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm_cell/Const_7?
lstm_cell/Mul_4Mullstm_cell/add_6:z:0lstm_cell/Const_6:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Mul_4?
lstm_cell/Add_7Addlstm_cell/Mul_4:z:0lstm_cell/Const_7:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Add_7?
#lstm_cell/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2%
#lstm_cell/clip_by_value_2/Minimum/y?
!lstm_cell/clip_by_value_2/MinimumMinimumlstm_cell/Add_7:z:0,lstm_cell/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:??????????2#
!lstm_cell/clip_by_value_2/Minimum
lstm_cell/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_cell/clip_by_value_2/y?
lstm_cell/clip_by_value_2Maximum%lstm_cell/clip_by_value_2/Minimum:z:0$lstm_cell/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/clip_by_value_2}
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_5:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid_1?
lstm_cell/mul_5Mullstm_cell/clip_by_value_2:z:0lstm_cell/Sigmoid_1:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_5?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0'lstm_cell_split_readvariableop_resource)lstm_cell_split_1_readvariableop_resource!lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_119531*
condR
while_cond_119530*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2
transpose_1?
IdentityIdentitystrided_slice_3:output:0^lstm_cell/ReadVariableOp^lstm_cell/ReadVariableOp_1^lstm_cell/ReadVariableOp_2^lstm_cell/ReadVariableOp_3^lstm_cell/split/ReadVariableOp!^lstm_cell/split_1/ReadVariableOp^while*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????:::24
lstm_cell/ReadVariableOplstm_cell/ReadVariableOp28
lstm_cell/ReadVariableOp_1lstm_cell/ReadVariableOp_128
lstm_cell/ReadVariableOp_2lstm_cell/ReadVariableOp_228
lstm_cell/ReadVariableOp_3lstm_cell/ReadVariableOp_32@
lstm_cell/split/ReadVariableOplstm_cell/split/ReadVariableOp2D
 lstm_cell/split_1/ReadVariableOp lstm_cell/split_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
F__inference_LSTM_lay_0_layer_call_and_return_conditional_losses_119941

inputs+
'lstm_cell_split_readvariableop_resource-
)lstm_cell_split_1_readvariableop_resource%
!lstm_cell_readvariableop_resource
identity??lstm_cell/ReadVariableOp?lstm_cell/ReadVariableOp_1?lstm_cell/ReadVariableOp_2?lstm_cell/ReadVariableOp_3?lstm_cell/split/ReadVariableOp? lstm_cell/split_1/ReadVariableOp?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:??????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_2d
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dim?
lstm_cell/split/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
lstm_cell/split/ReadVariableOp?
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
??:
??:
??:
??*
	num_split2
lstm_cell/split?
lstm_cell/MatMulMatMulstrided_slice_2:output:0lstm_cell/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul?
lstm_cell/MatMul_1MatMulstrided_slice_2:output:0lstm_cell/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_1?
lstm_cell/MatMul_2MatMulstrided_slice_2:output:0lstm_cell/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_2?
lstm_cell/MatMul_3MatMulstrided_slice_2:output:0lstm_cell/split:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_3h
lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Const_1|
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell/split_1/split_dim?
 lstm_cell/split_1/ReadVariableOpReadVariableOp)lstm_cell_split_1_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 lstm_cell/split_1/ReadVariableOp?
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2
lstm_cell/split_1?
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd?
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_1?
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_2?
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_3?
lstm_cell/ReadVariableOpReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp?
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
lstm_cell/strided_slice/stack?
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2!
lstm_cell/strided_slice/stack_1?
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
lstm_cell/strided_slice/stack_2?
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice?
lstm_cell/MatMul_4MatMulzeros:output:0 lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_4?
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/addk
lstm_cell/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
lstm_cell/Const_2k
lstm_cell/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm_cell/Const_3?
lstm_cell/MulMullstm_cell/add:z:0lstm_cell/Const_2:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Mul?
lstm_cell/Add_1Addlstm_cell/Mul:z:0lstm_cell/Const_3:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Add_1?
!lstm_cell/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2#
!lstm_cell/clip_by_value/Minimum/y?
lstm_cell/clip_by_value/MinimumMinimumlstm_cell/Add_1:z:0*lstm_cell/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:??????????2!
lstm_cell/clip_by_value/Minimum{
lstm_cell/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_cell/clip_by_value/y?
lstm_cell/clip_by_valueMaximum#lstm_cell/clip_by_value/Minimum:z:0"lstm_cell/clip_by_value/y:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/clip_by_value?
lstm_cell/ReadVariableOp_1ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_1?
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2!
lstm_cell/strided_slice_1/stack?
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_1/stack_1?
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_1/stack_2?
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_1?
lstm_cell/MatMul_5MatMulzeros:output:0"lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_5?
lstm_cell/add_2AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_2k
lstm_cell/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
lstm_cell/Const_4k
lstm_cell/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm_cell/Const_5?
lstm_cell/Mul_1Mullstm_cell/add_2:z:0lstm_cell/Const_4:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Mul_1?
lstm_cell/Add_3Addlstm_cell/Mul_1:z:0lstm_cell/Const_5:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Add_3?
#lstm_cell/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2%
#lstm_cell/clip_by_value_1/Minimum/y?
!lstm_cell/clip_by_value_1/MinimumMinimumlstm_cell/Add_3:z:0,lstm_cell/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:??????????2#
!lstm_cell/clip_by_value_1/Minimum
lstm_cell/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_cell/clip_by_value_1/y?
lstm_cell/clip_by_value_1Maximum%lstm_cell/clip_by_value_1/Minimum:z:0$lstm_cell/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/clip_by_value_1?
lstm_cell/mul_2Mullstm_cell/clip_by_value_1:z:0zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_2?
lstm_cell/ReadVariableOp_2ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_2?
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_2/stack?
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2#
!lstm_cell/strided_slice_2/stack_1?
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_2/stack_2?
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_2?
lstm_cell/MatMul_6MatMulzeros:output:0"lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_6?
lstm_cell/add_4AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_4y
lstm_cell/SigmoidSigmoidlstm_cell/add_4:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid?
lstm_cell/mul_3Mullstm_cell/clip_by_value:z:0lstm_cell/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_3?
lstm_cell/add_5AddV2lstm_cell/mul_2:z:0lstm_cell/mul_3:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_5?
lstm_cell/ReadVariableOp_3ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_3?
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2!
lstm_cell/strided_slice_3/stack?
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_3/stack_1?
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_3/stack_2?
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_3?
lstm_cell/MatMul_7MatMulzeros:output:0"lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_7?
lstm_cell/add_6AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_6k
lstm_cell/Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
lstm_cell/Const_6k
lstm_cell/Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm_cell/Const_7?
lstm_cell/Mul_4Mullstm_cell/add_6:z:0lstm_cell/Const_6:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Mul_4?
lstm_cell/Add_7Addlstm_cell/Mul_4:z:0lstm_cell/Const_7:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Add_7?
#lstm_cell/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2%
#lstm_cell/clip_by_value_2/Minimum/y?
!lstm_cell/clip_by_value_2/MinimumMinimumlstm_cell/Add_7:z:0,lstm_cell/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:??????????2#
!lstm_cell/clip_by_value_2/Minimum
lstm_cell/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_cell/clip_by_value_2/y?
lstm_cell/clip_by_value_2Maximum%lstm_cell/clip_by_value_2/Minimum:z:0$lstm_cell/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/clip_by_value_2}
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_5:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid_1?
lstm_cell/mul_5Mullstm_cell/clip_by_value_2:z:0lstm_cell/Sigmoid_1:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_5?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0'lstm_cell_split_readvariableop_resource)lstm_cell_split_1_readvariableop_resource!lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_119799*
condR
while_cond_119798*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2
transpose_1?
IdentityIdentitystrided_slice_3:output:0^lstm_cell/ReadVariableOp^lstm_cell/ReadVariableOp_1^lstm_cell/ReadVariableOp_2^lstm_cell/ReadVariableOp_3^lstm_cell/split/ReadVariableOp!^lstm_cell/split_1/ReadVariableOp^while*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????:::24
lstm_cell/ReadVariableOplstm_cell/ReadVariableOp28
lstm_cell/ReadVariableOp_1lstm_cell/ReadVariableOp_128
lstm_cell/ReadVariableOp_2lstm_cell/ReadVariableOp_228
lstm_cell/ReadVariableOp_3lstm_cell/ReadVariableOp_32@
lstm_cell/split/ReadVariableOplstm_cell/split/ReadVariableOp2D
 lstm_cell/split_1/ReadVariableOp lstm_cell/split_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
h
I__inference_Dropout_lay_0_layer_call_and_return_conditional_losses_121926

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
F__inference_Output_lay_layer_call_and_return_conditional_losses_120038

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?`*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????`2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:`*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????`2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????`2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
while_body_121482
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_03
/while_lstm_cell_split_readvariableop_resource_05
1while_lstm_cell_split_1_readvariableop_resource_0-
)while_lstm_cell_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor1
-while_lstm_cell_split_readvariableop_resource3
/while_lstm_cell_split_1_readvariableop_resource+
'while_lstm_cell_readvariableop_resource??while/lstm_cell/ReadVariableOp? while/lstm_cell/ReadVariableOp_1? while/lstm_cell/ReadVariableOp_2? while/lstm_cell/ReadVariableOp_3?$while/lstm_cell/split/ReadVariableOp?&while/lstm_cell/split_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemp
while/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell/Const?
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2!
while/lstm_cell/split/split_dim?
$while/lstm_cell/split/ReadVariableOpReadVariableOp/while_lstm_cell_split_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02&
$while/lstm_cell/split/ReadVariableOp?
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0,while/lstm_cell/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
??:
??:
??:
??*
	num_split2
while/lstm_cell/split?
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/MatMul?
while/lstm_cell/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:1*
T0*(
_output_shapes
:??????????2
while/lstm_cell/MatMul_1?
while/lstm_cell/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:2*
T0*(
_output_shapes
:??????????2
while/lstm_cell/MatMul_2?
while/lstm_cell/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:3*
T0*(
_output_shapes
:??????????2
while/lstm_cell/MatMul_3t
while/lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell/Const_1?
!while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!while/lstm_cell/split_1/split_dim?
&while/lstm_cell/split_1/ReadVariableOpReadVariableOp1while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
:?*
dtype02(
&while/lstm_cell/split_1/ReadVariableOp?
while/lstm_cell/split_1Split*while/lstm_cell/split_1/split_dim:output:0.while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2
while/lstm_cell/split_1?
while/lstm_cell/BiasAddBiasAdd while/lstm_cell/MatMul:product:0 while/lstm_cell/split_1:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/BiasAdd?
while/lstm_cell/BiasAdd_1BiasAdd"while/lstm_cell/MatMul_1:product:0 while/lstm_cell/split_1:output:1*
T0*(
_output_shapes
:??????????2
while/lstm_cell/BiasAdd_1?
while/lstm_cell/BiasAdd_2BiasAdd"while/lstm_cell/MatMul_2:product:0 while/lstm_cell/split_1:output:2*
T0*(
_output_shapes
:??????????2
while/lstm_cell/BiasAdd_2?
while/lstm_cell/BiasAdd_3BiasAdd"while/lstm_cell/MatMul_3:product:0 while/lstm_cell/split_1:output:3*
T0*(
_output_shapes
:??????????2
while/lstm_cell/BiasAdd_3?
while/lstm_cell/ReadVariableOpReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02 
while/lstm_cell/ReadVariableOp?
#while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2%
#while/lstm_cell/strided_slice/stack?
%while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2'
%while/lstm_cell/strided_slice/stack_1?
%while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2'
%while/lstm_cell/strided_slice/stack_2?
while/lstm_cell/strided_sliceStridedSlice&while/lstm_cell/ReadVariableOp:value:0,while/lstm_cell/strided_slice/stack:output:0.while/lstm_cell/strided_slice/stack_1:output:0.while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
while/lstm_cell/strided_slice?
while/lstm_cell/MatMul_4MatMulwhile_placeholder_2&while/lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/MatMul_4?
while/lstm_cell/addAddV2 while/lstm_cell/BiasAdd:output:0"while/lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/addw
while/lstm_cell/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/lstm_cell/Const_2w
while/lstm_cell/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/lstm_cell/Const_3?
while/lstm_cell/MulMulwhile/lstm_cell/add:z:0 while/lstm_cell/Const_2:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/Mul?
while/lstm_cell/Add_1Addwhile/lstm_cell/Mul:z:0 while/lstm_cell/Const_3:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/Add_1?
'while/lstm_cell/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2)
'while/lstm_cell/clip_by_value/Minimum/y?
%while/lstm_cell/clip_by_value/MinimumMinimumwhile/lstm_cell/Add_1:z:00while/lstm_cell/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:??????????2'
%while/lstm_cell/clip_by_value/Minimum?
while/lstm_cell/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
while/lstm_cell/clip_by_value/y?
while/lstm_cell/clip_by_valueMaximum)while/lstm_cell/clip_by_value/Minimum:z:0(while/lstm_cell/clip_by_value/y:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/clip_by_value?
 while/lstm_cell/ReadVariableOp_1ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02"
 while/lstm_cell/ReadVariableOp_1?
%while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2'
%while/lstm_cell/strided_slice_1/stack?
'while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2)
'while/lstm_cell/strided_slice_1/stack_1?
'while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_1/stack_2?
while/lstm_cell/strided_slice_1StridedSlice(while/lstm_cell/ReadVariableOp_1:value:0.while/lstm_cell/strided_slice_1/stack:output:00while/lstm_cell/strided_slice_1/stack_1:output:00while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2!
while/lstm_cell/strided_slice_1?
while/lstm_cell/MatMul_5MatMulwhile_placeholder_2(while/lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/MatMul_5?
while/lstm_cell/add_2AddV2"while/lstm_cell/BiasAdd_1:output:0"while/lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/add_2w
while/lstm_cell/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/lstm_cell/Const_4w
while/lstm_cell/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/lstm_cell/Const_5?
while/lstm_cell/Mul_1Mulwhile/lstm_cell/add_2:z:0 while/lstm_cell/Const_4:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/Mul_1?
while/lstm_cell/Add_3Addwhile/lstm_cell/Mul_1:z:0 while/lstm_cell/Const_5:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/Add_3?
)while/lstm_cell/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2+
)while/lstm_cell/clip_by_value_1/Minimum/y?
'while/lstm_cell/clip_by_value_1/MinimumMinimumwhile/lstm_cell/Add_3:z:02while/lstm_cell/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:??????????2)
'while/lstm_cell/clip_by_value_1/Minimum?
!while/lstm_cell/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!while/lstm_cell/clip_by_value_1/y?
while/lstm_cell/clip_by_value_1Maximum+while/lstm_cell/clip_by_value_1/Minimum:z:0*while/lstm_cell/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:??????????2!
while/lstm_cell/clip_by_value_1?
while/lstm_cell/mul_2Mul#while/lstm_cell/clip_by_value_1:z:0while_placeholder_3*
T0*(
_output_shapes
:??????????2
while/lstm_cell/mul_2?
 while/lstm_cell/ReadVariableOp_2ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02"
 while/lstm_cell/ReadVariableOp_2?
%while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2'
%while/lstm_cell/strided_slice_2/stack?
'while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2)
'while/lstm_cell/strided_slice_2/stack_1?
'while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_2/stack_2?
while/lstm_cell/strided_slice_2StridedSlice(while/lstm_cell/ReadVariableOp_2:value:0.while/lstm_cell/strided_slice_2/stack:output:00while/lstm_cell/strided_slice_2/stack_1:output:00while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2!
while/lstm_cell/strided_slice_2?
while/lstm_cell/MatMul_6MatMulwhile_placeholder_2(while/lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/MatMul_6?
while/lstm_cell/add_4AddV2"while/lstm_cell/BiasAdd_2:output:0"while/lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/add_4?
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/add_4:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/Sigmoid?
while/lstm_cell/mul_3Mul!while/lstm_cell/clip_by_value:z:0while/lstm_cell/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/mul_3?
while/lstm_cell/add_5AddV2while/lstm_cell/mul_2:z:0while/lstm_cell/mul_3:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/add_5?
 while/lstm_cell/ReadVariableOp_3ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02"
 while/lstm_cell/ReadVariableOp_3?
%while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2'
%while/lstm_cell/strided_slice_3/stack?
'while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'while/lstm_cell/strided_slice_3/stack_1?
'while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_3/stack_2?
while/lstm_cell/strided_slice_3StridedSlice(while/lstm_cell/ReadVariableOp_3:value:0.while/lstm_cell/strided_slice_3/stack:output:00while/lstm_cell/strided_slice_3/stack_1:output:00while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2!
while/lstm_cell/strided_slice_3?
while/lstm_cell/MatMul_7MatMulwhile_placeholder_2(while/lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/MatMul_7?
while/lstm_cell/add_6AddV2"while/lstm_cell/BiasAdd_3:output:0"while/lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/add_6w
while/lstm_cell/Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/lstm_cell/Const_6w
while/lstm_cell/Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/lstm_cell/Const_7?
while/lstm_cell/Mul_4Mulwhile/lstm_cell/add_6:z:0 while/lstm_cell/Const_6:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/Mul_4?
while/lstm_cell/Add_7Addwhile/lstm_cell/Mul_4:z:0 while/lstm_cell/Const_7:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/Add_7?
)while/lstm_cell/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2+
)while/lstm_cell/clip_by_value_2/Minimum/y?
'while/lstm_cell/clip_by_value_2/MinimumMinimumwhile/lstm_cell/Add_7:z:02while/lstm_cell/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:??????????2)
'while/lstm_cell/clip_by_value_2/Minimum?
!while/lstm_cell/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!while/lstm_cell/clip_by_value_2/y?
while/lstm_cell/clip_by_value_2Maximum+while/lstm_cell/clip_by_value_2/Minimum:z:0*while/lstm_cell/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:??????????2!
while/lstm_cell/clip_by_value_2?
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/add_5:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/Sigmoid_1?
while/lstm_cell/mul_5Mul#while/lstm_cell/clip_by_value_2:z:0while/lstm_cell/Sigmoid_1:y:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/mul_5?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_5:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:0^while/lstm_cell/ReadVariableOp!^while/lstm_cell/ReadVariableOp_1!^while/lstm_cell/ReadVariableOp_2!^while/lstm_cell/ReadVariableOp_3%^while/lstm_cell/split/ReadVariableOp'^while/lstm_cell/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations^while/lstm_cell/ReadVariableOp!^while/lstm_cell/ReadVariableOp_1!^while/lstm_cell/ReadVariableOp_2!^while/lstm_cell/ReadVariableOp_3%^while/lstm_cell/split/ReadVariableOp'^while/lstm_cell/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0^while/lstm_cell/ReadVariableOp!^while/lstm_cell/ReadVariableOp_1!^while/lstm_cell/ReadVariableOp_2!^while/lstm_cell/ReadVariableOp_3%^while/lstm_cell/split/ReadVariableOp'^while/lstm_cell/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/lstm_cell/ReadVariableOp!^while/lstm_cell/ReadVariableOp_1!^while/lstm_cell/ReadVariableOp_2!^while/lstm_cell/ReadVariableOp_3%^while/lstm_cell/split/ReadVariableOp'^while/lstm_cell/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell/mul_5:z:0^while/lstm_cell/ReadVariableOp!^while/lstm_cell/ReadVariableOp_1!^while/lstm_cell/ReadVariableOp_2!^while/lstm_cell/ReadVariableOp_3%^while/lstm_cell/split/ReadVariableOp'^while/lstm_cell/split_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell/add_5:z:0^while/lstm_cell/ReadVariableOp!^while/lstm_cell/ReadVariableOp_1!^while/lstm_cell/ReadVariableOp_2!^while/lstm_cell/ReadVariableOp_3%^while/lstm_cell/split/ReadVariableOp'^while/lstm_cell/split_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"T
'while_lstm_cell_readvariableop_resource)while_lstm_cell_readvariableop_resource_0"d
/while_lstm_cell_split_1_readvariableop_resource1while_lstm_cell_split_1_readvariableop_resource_0"`
-while_lstm_cell_split_readvariableop_resource/while_lstm_cell_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :??????????:??????????: : :::2@
while/lstm_cell/ReadVariableOpwhile/lstm_cell/ReadVariableOp2D
 while/lstm_cell/ReadVariableOp_1 while/lstm_cell/ReadVariableOp_12D
 while/lstm_cell/ReadVariableOp_2 while/lstm_cell/ReadVariableOp_22D
 while/lstm_cell/ReadVariableOp_3 while/lstm_cell/ReadVariableOp_32L
$while/lstm_cell/split/ReadVariableOp$while/lstm_cell/split/ReadVariableOp2P
&while/lstm_cell/split_1/ReadVariableOp&while/lstm_cell/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
*__inference_lstm_cell_layer_call_fn_122196

inputs
states_0
states_1
unknown
	unknown_0
	unknown_1
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_lstm_cell_layer_call_and_return_conditional_losses_1189022
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*[
_input_shapesJ
H:??????????:??????????:??????????:::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/1
?
?
F__inference_sequential_layer_call_and_return_conditional_losses_120077
lstm_lay_0_input
lstm_lay_0_120058
lstm_lay_0_120060
lstm_lay_0_120062
dense_lay_1_120066
dense_lay_1_120068
output_lay_120071
output_lay_120073
identity??#Dense_lay_1/StatefulPartitionedCall?"LSTM_lay_0/StatefulPartitionedCall?"Output_lay/StatefulPartitionedCall?
"LSTM_lay_0/StatefulPartitionedCallStatefulPartitionedCalllstm_lay_0_inputlstm_lay_0_120058lstm_lay_0_120060lstm_lay_0_120062*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_LSTM_lay_0_layer_call_and_return_conditional_losses_1199412$
"LSTM_lay_0/StatefulPartitionedCall?
Dropout_lay_0/PartitionedCallPartitionedCall+LSTM_lay_0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_Dropout_lay_0_layer_call_and_return_conditional_losses_1199882
Dropout_lay_0/PartitionedCall?
#Dense_lay_1/StatefulPartitionedCallStatefulPartitionedCall&Dropout_lay_0/PartitionedCall:output:0dense_lay_1_120066dense_lay_1_120068*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_Dense_lay_1_layer_call_and_return_conditional_losses_1200122%
#Dense_lay_1/StatefulPartitionedCall?
"Output_lay/StatefulPartitionedCallStatefulPartitionedCall,Dense_lay_1/StatefulPartitionedCall:output:0output_lay_120071output_lay_120073*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????`*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_Output_lay_layer_call_and_return_conditional_losses_1200382$
"Output_lay/StatefulPartitionedCall?
IdentityIdentity+Output_lay/StatefulPartitionedCall:output:0$^Dense_lay_1/StatefulPartitionedCall#^LSTM_lay_0/StatefulPartitionedCall#^Output_lay/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????`2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:??????????:::::::2J
#Dense_lay_1/StatefulPartitionedCall#Dense_lay_1/StatefulPartitionedCall2H
"LSTM_lay_0/StatefulPartitionedCall"LSTM_lay_0/StatefulPartitionedCall2H
"Output_lay/StatefulPartitionedCall"Output_lay/StatefulPartitionedCall:^ Z
,
_output_shapes
:??????????
*
_user_specified_nameLSTM_lay_0_input
??
?
while_body_121192
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_03
/while_lstm_cell_split_readvariableop_resource_05
1while_lstm_cell_split_1_readvariableop_resource_0-
)while_lstm_cell_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor1
-while_lstm_cell_split_readvariableop_resource3
/while_lstm_cell_split_1_readvariableop_resource+
'while_lstm_cell_readvariableop_resource??while/lstm_cell/ReadVariableOp? while/lstm_cell/ReadVariableOp_1? while/lstm_cell/ReadVariableOp_2? while/lstm_cell/ReadVariableOp_3?$while/lstm_cell/split/ReadVariableOp?&while/lstm_cell/split_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemp
while/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell/Const?
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2!
while/lstm_cell/split/split_dim?
$while/lstm_cell/split/ReadVariableOpReadVariableOp/while_lstm_cell_split_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02&
$while/lstm_cell/split/ReadVariableOp?
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0,while/lstm_cell/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
??:
??:
??:
??*
	num_split2
while/lstm_cell/split?
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/MatMul?
while/lstm_cell/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:1*
T0*(
_output_shapes
:??????????2
while/lstm_cell/MatMul_1?
while/lstm_cell/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:2*
T0*(
_output_shapes
:??????????2
while/lstm_cell/MatMul_2?
while/lstm_cell/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:3*
T0*(
_output_shapes
:??????????2
while/lstm_cell/MatMul_3t
while/lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell/Const_1?
!while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!while/lstm_cell/split_1/split_dim?
&while/lstm_cell/split_1/ReadVariableOpReadVariableOp1while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
:?*
dtype02(
&while/lstm_cell/split_1/ReadVariableOp?
while/lstm_cell/split_1Split*while/lstm_cell/split_1/split_dim:output:0.while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2
while/lstm_cell/split_1?
while/lstm_cell/BiasAddBiasAdd while/lstm_cell/MatMul:product:0 while/lstm_cell/split_1:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/BiasAdd?
while/lstm_cell/BiasAdd_1BiasAdd"while/lstm_cell/MatMul_1:product:0 while/lstm_cell/split_1:output:1*
T0*(
_output_shapes
:??????????2
while/lstm_cell/BiasAdd_1?
while/lstm_cell/BiasAdd_2BiasAdd"while/lstm_cell/MatMul_2:product:0 while/lstm_cell/split_1:output:2*
T0*(
_output_shapes
:??????????2
while/lstm_cell/BiasAdd_2?
while/lstm_cell/BiasAdd_3BiasAdd"while/lstm_cell/MatMul_3:product:0 while/lstm_cell/split_1:output:3*
T0*(
_output_shapes
:??????????2
while/lstm_cell/BiasAdd_3?
while/lstm_cell/ReadVariableOpReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02 
while/lstm_cell/ReadVariableOp?
#while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2%
#while/lstm_cell/strided_slice/stack?
%while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2'
%while/lstm_cell/strided_slice/stack_1?
%while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2'
%while/lstm_cell/strided_slice/stack_2?
while/lstm_cell/strided_sliceStridedSlice&while/lstm_cell/ReadVariableOp:value:0,while/lstm_cell/strided_slice/stack:output:0.while/lstm_cell/strided_slice/stack_1:output:0.while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
while/lstm_cell/strided_slice?
while/lstm_cell/MatMul_4MatMulwhile_placeholder_2&while/lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/MatMul_4?
while/lstm_cell/addAddV2 while/lstm_cell/BiasAdd:output:0"while/lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/addw
while/lstm_cell/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/lstm_cell/Const_2w
while/lstm_cell/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/lstm_cell/Const_3?
while/lstm_cell/MulMulwhile/lstm_cell/add:z:0 while/lstm_cell/Const_2:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/Mul?
while/lstm_cell/Add_1Addwhile/lstm_cell/Mul:z:0 while/lstm_cell/Const_3:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/Add_1?
'while/lstm_cell/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2)
'while/lstm_cell/clip_by_value/Minimum/y?
%while/lstm_cell/clip_by_value/MinimumMinimumwhile/lstm_cell/Add_1:z:00while/lstm_cell/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:??????????2'
%while/lstm_cell/clip_by_value/Minimum?
while/lstm_cell/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
while/lstm_cell/clip_by_value/y?
while/lstm_cell/clip_by_valueMaximum)while/lstm_cell/clip_by_value/Minimum:z:0(while/lstm_cell/clip_by_value/y:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/clip_by_value?
 while/lstm_cell/ReadVariableOp_1ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02"
 while/lstm_cell/ReadVariableOp_1?
%while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2'
%while/lstm_cell/strided_slice_1/stack?
'while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2)
'while/lstm_cell/strided_slice_1/stack_1?
'while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_1/stack_2?
while/lstm_cell/strided_slice_1StridedSlice(while/lstm_cell/ReadVariableOp_1:value:0.while/lstm_cell/strided_slice_1/stack:output:00while/lstm_cell/strided_slice_1/stack_1:output:00while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2!
while/lstm_cell/strided_slice_1?
while/lstm_cell/MatMul_5MatMulwhile_placeholder_2(while/lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/MatMul_5?
while/lstm_cell/add_2AddV2"while/lstm_cell/BiasAdd_1:output:0"while/lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/add_2w
while/lstm_cell/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/lstm_cell/Const_4w
while/lstm_cell/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/lstm_cell/Const_5?
while/lstm_cell/Mul_1Mulwhile/lstm_cell/add_2:z:0 while/lstm_cell/Const_4:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/Mul_1?
while/lstm_cell/Add_3Addwhile/lstm_cell/Mul_1:z:0 while/lstm_cell/Const_5:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/Add_3?
)while/lstm_cell/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2+
)while/lstm_cell/clip_by_value_1/Minimum/y?
'while/lstm_cell/clip_by_value_1/MinimumMinimumwhile/lstm_cell/Add_3:z:02while/lstm_cell/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:??????????2)
'while/lstm_cell/clip_by_value_1/Minimum?
!while/lstm_cell/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!while/lstm_cell/clip_by_value_1/y?
while/lstm_cell/clip_by_value_1Maximum+while/lstm_cell/clip_by_value_1/Minimum:z:0*while/lstm_cell/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:??????????2!
while/lstm_cell/clip_by_value_1?
while/lstm_cell/mul_2Mul#while/lstm_cell/clip_by_value_1:z:0while_placeholder_3*
T0*(
_output_shapes
:??????????2
while/lstm_cell/mul_2?
 while/lstm_cell/ReadVariableOp_2ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02"
 while/lstm_cell/ReadVariableOp_2?
%while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2'
%while/lstm_cell/strided_slice_2/stack?
'while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2)
'while/lstm_cell/strided_slice_2/stack_1?
'while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_2/stack_2?
while/lstm_cell/strided_slice_2StridedSlice(while/lstm_cell/ReadVariableOp_2:value:0.while/lstm_cell/strided_slice_2/stack:output:00while/lstm_cell/strided_slice_2/stack_1:output:00while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2!
while/lstm_cell/strided_slice_2?
while/lstm_cell/MatMul_6MatMulwhile_placeholder_2(while/lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/MatMul_6?
while/lstm_cell/add_4AddV2"while/lstm_cell/BiasAdd_2:output:0"while/lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/add_4?
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/add_4:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/Sigmoid?
while/lstm_cell/mul_3Mul!while/lstm_cell/clip_by_value:z:0while/lstm_cell/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/mul_3?
while/lstm_cell/add_5AddV2while/lstm_cell/mul_2:z:0while/lstm_cell/mul_3:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/add_5?
 while/lstm_cell/ReadVariableOp_3ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02"
 while/lstm_cell/ReadVariableOp_3?
%while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2'
%while/lstm_cell/strided_slice_3/stack?
'while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'while/lstm_cell/strided_slice_3/stack_1?
'while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_3/stack_2?
while/lstm_cell/strided_slice_3StridedSlice(while/lstm_cell/ReadVariableOp_3:value:0.while/lstm_cell/strided_slice_3/stack:output:00while/lstm_cell/strided_slice_3/stack_1:output:00while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2!
while/lstm_cell/strided_slice_3?
while/lstm_cell/MatMul_7MatMulwhile_placeholder_2(while/lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/MatMul_7?
while/lstm_cell/add_6AddV2"while/lstm_cell/BiasAdd_3:output:0"while/lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/add_6w
while/lstm_cell/Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/lstm_cell/Const_6w
while/lstm_cell/Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/lstm_cell/Const_7?
while/lstm_cell/Mul_4Mulwhile/lstm_cell/add_6:z:0 while/lstm_cell/Const_6:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/Mul_4?
while/lstm_cell/Add_7Addwhile/lstm_cell/Mul_4:z:0 while/lstm_cell/Const_7:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/Add_7?
)while/lstm_cell/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2+
)while/lstm_cell/clip_by_value_2/Minimum/y?
'while/lstm_cell/clip_by_value_2/MinimumMinimumwhile/lstm_cell/Add_7:z:02while/lstm_cell/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:??????????2)
'while/lstm_cell/clip_by_value_2/Minimum?
!while/lstm_cell/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!while/lstm_cell/clip_by_value_2/y?
while/lstm_cell/clip_by_value_2Maximum+while/lstm_cell/clip_by_value_2/Minimum:z:0*while/lstm_cell/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:??????????2!
while/lstm_cell/clip_by_value_2?
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/add_5:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/Sigmoid_1?
while/lstm_cell/mul_5Mul#while/lstm_cell/clip_by_value_2:z:0while/lstm_cell/Sigmoid_1:y:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/mul_5?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_5:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:0^while/lstm_cell/ReadVariableOp!^while/lstm_cell/ReadVariableOp_1!^while/lstm_cell/ReadVariableOp_2!^while/lstm_cell/ReadVariableOp_3%^while/lstm_cell/split/ReadVariableOp'^while/lstm_cell/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations^while/lstm_cell/ReadVariableOp!^while/lstm_cell/ReadVariableOp_1!^while/lstm_cell/ReadVariableOp_2!^while/lstm_cell/ReadVariableOp_3%^while/lstm_cell/split/ReadVariableOp'^while/lstm_cell/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0^while/lstm_cell/ReadVariableOp!^while/lstm_cell/ReadVariableOp_1!^while/lstm_cell/ReadVariableOp_2!^while/lstm_cell/ReadVariableOp_3%^while/lstm_cell/split/ReadVariableOp'^while/lstm_cell/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/lstm_cell/ReadVariableOp!^while/lstm_cell/ReadVariableOp_1!^while/lstm_cell/ReadVariableOp_2!^while/lstm_cell/ReadVariableOp_3%^while/lstm_cell/split/ReadVariableOp'^while/lstm_cell/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell/mul_5:z:0^while/lstm_cell/ReadVariableOp!^while/lstm_cell/ReadVariableOp_1!^while/lstm_cell/ReadVariableOp_2!^while/lstm_cell/ReadVariableOp_3%^while/lstm_cell/split/ReadVariableOp'^while/lstm_cell/split_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell/add_5:z:0^while/lstm_cell/ReadVariableOp!^while/lstm_cell/ReadVariableOp_1!^while/lstm_cell/ReadVariableOp_2!^while/lstm_cell/ReadVariableOp_3%^while/lstm_cell/split/ReadVariableOp'^while/lstm_cell/split_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"T
'while_lstm_cell_readvariableop_resource)while_lstm_cell_readvariableop_resource_0"d
/while_lstm_cell_split_1_readvariableop_resource1while_lstm_cell_split_1_readvariableop_resource_0"`
-while_lstm_cell_split_readvariableop_resource/while_lstm_cell_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :??????????:??????????: : :::2@
while/lstm_cell/ReadVariableOpwhile/lstm_cell/ReadVariableOp2D
 while/lstm_cell/ReadVariableOp_1 while/lstm_cell/ReadVariableOp_12D
 while/lstm_cell/ReadVariableOp_2 while/lstm_cell/ReadVariableOp_22D
 while/lstm_cell/ReadVariableOp_3 while/lstm_cell/ReadVariableOp_32L
$while/lstm_cell/split/ReadVariableOp$while/lstm_cell/split/ReadVariableOp2P
&while/lstm_cell/split_1/ReadVariableOp&while/lstm_cell/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?C
?
F__inference_LSTM_lay_0_layer_call_and_return_conditional_losses_119393

inputs
lstm_cell_119312
lstm_cell_119314
lstm_cell_119316
identity??!lstm_cell/StatefulPartitionedCall?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_2?
!lstm_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_119312lstm_cell_119314lstm_cell_119316*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_lstm_cell_layer_call_and_return_conditional_losses_1189022#
!lstm_cell/StatefulPartitionedCall?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_119312lstm_cell_119314lstm_cell_119316*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_119325*
condR
while_cond_119324*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
transpose_1?
IdentityIdentitystrided_slice_3:output:0"^lstm_cell/StatefulPartitionedCall^while*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:???????????????????:::2F
!lstm_cell/StatefulPartitionedCall!lstm_cell/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
??
?
!__inference__wrapped_model_118680
lstm_lay_0_inputA
=sequential_lstm_lay_0_lstm_cell_split_readvariableop_resourceC
?sequential_lstm_lay_0_lstm_cell_split_1_readvariableop_resource;
7sequential_lstm_lay_0_lstm_cell_readvariableop_resource9
5sequential_dense_lay_1_matmul_readvariableop_resource:
6sequential_dense_lay_1_biasadd_readvariableop_resource8
4sequential_output_lay_matmul_readvariableop_resource9
5sequential_output_lay_biasadd_readvariableop_resource
identity??-sequential/Dense_lay_1/BiasAdd/ReadVariableOp?,sequential/Dense_lay_1/MatMul/ReadVariableOp?.sequential/LSTM_lay_0/lstm_cell/ReadVariableOp?0sequential/LSTM_lay_0/lstm_cell/ReadVariableOp_1?0sequential/LSTM_lay_0/lstm_cell/ReadVariableOp_2?0sequential/LSTM_lay_0/lstm_cell/ReadVariableOp_3?4sequential/LSTM_lay_0/lstm_cell/split/ReadVariableOp?6sequential/LSTM_lay_0/lstm_cell/split_1/ReadVariableOp?sequential/LSTM_lay_0/while?,sequential/Output_lay/BiasAdd/ReadVariableOp?+sequential/Output_lay/MatMul/ReadVariableOpz
sequential/LSTM_lay_0/ShapeShapelstm_lay_0_input*
T0*
_output_shapes
:2
sequential/LSTM_lay_0/Shape?
)sequential/LSTM_lay_0/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential/LSTM_lay_0/strided_slice/stack?
+sequential/LSTM_lay_0/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential/LSTM_lay_0/strided_slice/stack_1?
+sequential/LSTM_lay_0/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential/LSTM_lay_0/strided_slice/stack_2?
#sequential/LSTM_lay_0/strided_sliceStridedSlice$sequential/LSTM_lay_0/Shape:output:02sequential/LSTM_lay_0/strided_slice/stack:output:04sequential/LSTM_lay_0/strided_slice/stack_1:output:04sequential/LSTM_lay_0/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#sequential/LSTM_lay_0/strided_slice?
!sequential/LSTM_lay_0/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2#
!sequential/LSTM_lay_0/zeros/mul/y?
sequential/LSTM_lay_0/zeros/mulMul,sequential/LSTM_lay_0/strided_slice:output:0*sequential/LSTM_lay_0/zeros/mul/y:output:0*
T0*
_output_shapes
: 2!
sequential/LSTM_lay_0/zeros/mul?
"sequential/LSTM_lay_0/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2$
"sequential/LSTM_lay_0/zeros/Less/y?
 sequential/LSTM_lay_0/zeros/LessLess#sequential/LSTM_lay_0/zeros/mul:z:0+sequential/LSTM_lay_0/zeros/Less/y:output:0*
T0*
_output_shapes
: 2"
 sequential/LSTM_lay_0/zeros/Less?
$sequential/LSTM_lay_0/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2&
$sequential/LSTM_lay_0/zeros/packed/1?
"sequential/LSTM_lay_0/zeros/packedPack,sequential/LSTM_lay_0/strided_slice:output:0-sequential/LSTM_lay_0/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2$
"sequential/LSTM_lay_0/zeros/packed?
!sequential/LSTM_lay_0/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!sequential/LSTM_lay_0/zeros/Const?
sequential/LSTM_lay_0/zerosFill+sequential/LSTM_lay_0/zeros/packed:output:0*sequential/LSTM_lay_0/zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
sequential/LSTM_lay_0/zeros?
#sequential/LSTM_lay_0/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2%
#sequential/LSTM_lay_0/zeros_1/mul/y?
!sequential/LSTM_lay_0/zeros_1/mulMul,sequential/LSTM_lay_0/strided_slice:output:0,sequential/LSTM_lay_0/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2#
!sequential/LSTM_lay_0/zeros_1/mul?
$sequential/LSTM_lay_0/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2&
$sequential/LSTM_lay_0/zeros_1/Less/y?
"sequential/LSTM_lay_0/zeros_1/LessLess%sequential/LSTM_lay_0/zeros_1/mul:z:0-sequential/LSTM_lay_0/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2$
"sequential/LSTM_lay_0/zeros_1/Less?
&sequential/LSTM_lay_0/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2(
&sequential/LSTM_lay_0/zeros_1/packed/1?
$sequential/LSTM_lay_0/zeros_1/packedPack,sequential/LSTM_lay_0/strided_slice:output:0/sequential/LSTM_lay_0/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2&
$sequential/LSTM_lay_0/zeros_1/packed?
#sequential/LSTM_lay_0/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#sequential/LSTM_lay_0/zeros_1/Const?
sequential/LSTM_lay_0/zeros_1Fill-sequential/LSTM_lay_0/zeros_1/packed:output:0,sequential/LSTM_lay_0/zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2
sequential/LSTM_lay_0/zeros_1?
$sequential/LSTM_lay_0/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2&
$sequential/LSTM_lay_0/transpose/perm?
sequential/LSTM_lay_0/transpose	Transposelstm_lay_0_input-sequential/LSTM_lay_0/transpose/perm:output:0*
T0*,
_output_shapes
:??????????2!
sequential/LSTM_lay_0/transpose?
sequential/LSTM_lay_0/Shape_1Shape#sequential/LSTM_lay_0/transpose:y:0*
T0*
_output_shapes
:2
sequential/LSTM_lay_0/Shape_1?
+sequential/LSTM_lay_0/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential/LSTM_lay_0/strided_slice_1/stack?
-sequential/LSTM_lay_0/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/LSTM_lay_0/strided_slice_1/stack_1?
-sequential/LSTM_lay_0/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/LSTM_lay_0/strided_slice_1/stack_2?
%sequential/LSTM_lay_0/strided_slice_1StridedSlice&sequential/LSTM_lay_0/Shape_1:output:04sequential/LSTM_lay_0/strided_slice_1/stack:output:06sequential/LSTM_lay_0/strided_slice_1/stack_1:output:06sequential/LSTM_lay_0/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%sequential/LSTM_lay_0/strided_slice_1?
1sequential/LSTM_lay_0/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????23
1sequential/LSTM_lay_0/TensorArrayV2/element_shape?
#sequential/LSTM_lay_0/TensorArrayV2TensorListReserve:sequential/LSTM_lay_0/TensorArrayV2/element_shape:output:0.sequential/LSTM_lay_0/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02%
#sequential/LSTM_lay_0/TensorArrayV2?
Ksequential/LSTM_lay_0/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2M
Ksequential/LSTM_lay_0/TensorArrayUnstack/TensorListFromTensor/element_shape?
=sequential/LSTM_lay_0/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#sequential/LSTM_lay_0/transpose:y:0Tsequential/LSTM_lay_0/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02?
=sequential/LSTM_lay_0/TensorArrayUnstack/TensorListFromTensor?
+sequential/LSTM_lay_0/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential/LSTM_lay_0/strided_slice_2/stack?
-sequential/LSTM_lay_0/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/LSTM_lay_0/strided_slice_2/stack_1?
-sequential/LSTM_lay_0/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/LSTM_lay_0/strided_slice_2/stack_2?
%sequential/LSTM_lay_0/strided_slice_2StridedSlice#sequential/LSTM_lay_0/transpose:y:04sequential/LSTM_lay_0/strided_slice_2/stack:output:06sequential/LSTM_lay_0/strided_slice_2/stack_1:output:06sequential/LSTM_lay_0/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2'
%sequential/LSTM_lay_0/strided_slice_2?
%sequential/LSTM_lay_0/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2'
%sequential/LSTM_lay_0/lstm_cell/Const?
/sequential/LSTM_lay_0/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :21
/sequential/LSTM_lay_0/lstm_cell/split/split_dim?
4sequential/LSTM_lay_0/lstm_cell/split/ReadVariableOpReadVariableOp=sequential_lstm_lay_0_lstm_cell_split_readvariableop_resource* 
_output_shapes
:
??*
dtype026
4sequential/LSTM_lay_0/lstm_cell/split/ReadVariableOp?
%sequential/LSTM_lay_0/lstm_cell/splitSplit8sequential/LSTM_lay_0/lstm_cell/split/split_dim:output:0<sequential/LSTM_lay_0/lstm_cell/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
??:
??:
??:
??*
	num_split2'
%sequential/LSTM_lay_0/lstm_cell/split?
&sequential/LSTM_lay_0/lstm_cell/MatMulMatMul.sequential/LSTM_lay_0/strided_slice_2:output:0.sequential/LSTM_lay_0/lstm_cell/split:output:0*
T0*(
_output_shapes
:??????????2(
&sequential/LSTM_lay_0/lstm_cell/MatMul?
(sequential/LSTM_lay_0/lstm_cell/MatMul_1MatMul.sequential/LSTM_lay_0/strided_slice_2:output:0.sequential/LSTM_lay_0/lstm_cell/split:output:1*
T0*(
_output_shapes
:??????????2*
(sequential/LSTM_lay_0/lstm_cell/MatMul_1?
(sequential/LSTM_lay_0/lstm_cell/MatMul_2MatMul.sequential/LSTM_lay_0/strided_slice_2:output:0.sequential/LSTM_lay_0/lstm_cell/split:output:2*
T0*(
_output_shapes
:??????????2*
(sequential/LSTM_lay_0/lstm_cell/MatMul_2?
(sequential/LSTM_lay_0/lstm_cell/MatMul_3MatMul.sequential/LSTM_lay_0/strided_slice_2:output:0.sequential/LSTM_lay_0/lstm_cell/split:output:3*
T0*(
_output_shapes
:??????????2*
(sequential/LSTM_lay_0/lstm_cell/MatMul_3?
'sequential/LSTM_lay_0/lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential/LSTM_lay_0/lstm_cell/Const_1?
1sequential/LSTM_lay_0/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 23
1sequential/LSTM_lay_0/lstm_cell/split_1/split_dim?
6sequential/LSTM_lay_0/lstm_cell/split_1/ReadVariableOpReadVariableOp?sequential_lstm_lay_0_lstm_cell_split_1_readvariableop_resource*
_output_shapes	
:?*
dtype028
6sequential/LSTM_lay_0/lstm_cell/split_1/ReadVariableOp?
'sequential/LSTM_lay_0/lstm_cell/split_1Split:sequential/LSTM_lay_0/lstm_cell/split_1/split_dim:output:0>sequential/LSTM_lay_0/lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2)
'sequential/LSTM_lay_0/lstm_cell/split_1?
'sequential/LSTM_lay_0/lstm_cell/BiasAddBiasAdd0sequential/LSTM_lay_0/lstm_cell/MatMul:product:00sequential/LSTM_lay_0/lstm_cell/split_1:output:0*
T0*(
_output_shapes
:??????????2)
'sequential/LSTM_lay_0/lstm_cell/BiasAdd?
)sequential/LSTM_lay_0/lstm_cell/BiasAdd_1BiasAdd2sequential/LSTM_lay_0/lstm_cell/MatMul_1:product:00sequential/LSTM_lay_0/lstm_cell/split_1:output:1*
T0*(
_output_shapes
:??????????2+
)sequential/LSTM_lay_0/lstm_cell/BiasAdd_1?
)sequential/LSTM_lay_0/lstm_cell/BiasAdd_2BiasAdd2sequential/LSTM_lay_0/lstm_cell/MatMul_2:product:00sequential/LSTM_lay_0/lstm_cell/split_1:output:2*
T0*(
_output_shapes
:??????????2+
)sequential/LSTM_lay_0/lstm_cell/BiasAdd_2?
)sequential/LSTM_lay_0/lstm_cell/BiasAdd_3BiasAdd2sequential/LSTM_lay_0/lstm_cell/MatMul_3:product:00sequential/LSTM_lay_0/lstm_cell/split_1:output:3*
T0*(
_output_shapes
:??????????2+
)sequential/LSTM_lay_0/lstm_cell/BiasAdd_3?
.sequential/LSTM_lay_0/lstm_cell/ReadVariableOpReadVariableOp7sequential_lstm_lay_0_lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype020
.sequential/LSTM_lay_0/lstm_cell/ReadVariableOp?
3sequential/LSTM_lay_0/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        25
3sequential/LSTM_lay_0/lstm_cell/strided_slice/stack?
5sequential/LSTM_lay_0/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   27
5sequential/LSTM_lay_0/lstm_cell/strided_slice/stack_1?
5sequential/LSTM_lay_0/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      27
5sequential/LSTM_lay_0/lstm_cell/strided_slice/stack_2?
-sequential/LSTM_lay_0/lstm_cell/strided_sliceStridedSlice6sequential/LSTM_lay_0/lstm_cell/ReadVariableOp:value:0<sequential/LSTM_lay_0/lstm_cell/strided_slice/stack:output:0>sequential/LSTM_lay_0/lstm_cell/strided_slice/stack_1:output:0>sequential/LSTM_lay_0/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2/
-sequential/LSTM_lay_0/lstm_cell/strided_slice?
(sequential/LSTM_lay_0/lstm_cell/MatMul_4MatMul$sequential/LSTM_lay_0/zeros:output:06sequential/LSTM_lay_0/lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:??????????2*
(sequential/LSTM_lay_0/lstm_cell/MatMul_4?
#sequential/LSTM_lay_0/lstm_cell/addAddV20sequential/LSTM_lay_0/lstm_cell/BiasAdd:output:02sequential/LSTM_lay_0/lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2%
#sequential/LSTM_lay_0/lstm_cell/add?
'sequential/LSTM_lay_0/lstm_cell/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2)
'sequential/LSTM_lay_0/lstm_cell/Const_2?
'sequential/LSTM_lay_0/lstm_cell/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2)
'sequential/LSTM_lay_0/lstm_cell/Const_3?
#sequential/LSTM_lay_0/lstm_cell/MulMul'sequential/LSTM_lay_0/lstm_cell/add:z:00sequential/LSTM_lay_0/lstm_cell/Const_2:output:0*
T0*(
_output_shapes
:??????????2%
#sequential/LSTM_lay_0/lstm_cell/Mul?
%sequential/LSTM_lay_0/lstm_cell/Add_1Add'sequential/LSTM_lay_0/lstm_cell/Mul:z:00sequential/LSTM_lay_0/lstm_cell/Const_3:output:0*
T0*(
_output_shapes
:??????????2'
%sequential/LSTM_lay_0/lstm_cell/Add_1?
7sequential/LSTM_lay_0/lstm_cell/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??29
7sequential/LSTM_lay_0/lstm_cell/clip_by_value/Minimum/y?
5sequential/LSTM_lay_0/lstm_cell/clip_by_value/MinimumMinimum)sequential/LSTM_lay_0/lstm_cell/Add_1:z:0@sequential/LSTM_lay_0/lstm_cell/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:??????????27
5sequential/LSTM_lay_0/lstm_cell/clip_by_value/Minimum?
/sequential/LSTM_lay_0/lstm_cell/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    21
/sequential/LSTM_lay_0/lstm_cell/clip_by_value/y?
-sequential/LSTM_lay_0/lstm_cell/clip_by_valueMaximum9sequential/LSTM_lay_0/lstm_cell/clip_by_value/Minimum:z:08sequential/LSTM_lay_0/lstm_cell/clip_by_value/y:output:0*
T0*(
_output_shapes
:??????????2/
-sequential/LSTM_lay_0/lstm_cell/clip_by_value?
0sequential/LSTM_lay_0/lstm_cell/ReadVariableOp_1ReadVariableOp7sequential_lstm_lay_0_lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype022
0sequential/LSTM_lay_0/lstm_cell/ReadVariableOp_1?
5sequential/LSTM_lay_0/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   27
5sequential/LSTM_lay_0/lstm_cell/strided_slice_1/stack?
7sequential/LSTM_lay_0/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       29
7sequential/LSTM_lay_0/lstm_cell/strided_slice_1/stack_1?
7sequential/LSTM_lay_0/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7sequential/LSTM_lay_0/lstm_cell/strided_slice_1/stack_2?
/sequential/LSTM_lay_0/lstm_cell/strided_slice_1StridedSlice8sequential/LSTM_lay_0/lstm_cell/ReadVariableOp_1:value:0>sequential/LSTM_lay_0/lstm_cell/strided_slice_1/stack:output:0@sequential/LSTM_lay_0/lstm_cell/strided_slice_1/stack_1:output:0@sequential/LSTM_lay_0/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask21
/sequential/LSTM_lay_0/lstm_cell/strided_slice_1?
(sequential/LSTM_lay_0/lstm_cell/MatMul_5MatMul$sequential/LSTM_lay_0/zeros:output:08sequential/LSTM_lay_0/lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2*
(sequential/LSTM_lay_0/lstm_cell/MatMul_5?
%sequential/LSTM_lay_0/lstm_cell/add_2AddV22sequential/LSTM_lay_0/lstm_cell/BiasAdd_1:output:02sequential/LSTM_lay_0/lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2'
%sequential/LSTM_lay_0/lstm_cell/add_2?
'sequential/LSTM_lay_0/lstm_cell/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2)
'sequential/LSTM_lay_0/lstm_cell/Const_4?
'sequential/LSTM_lay_0/lstm_cell/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2)
'sequential/LSTM_lay_0/lstm_cell/Const_5?
%sequential/LSTM_lay_0/lstm_cell/Mul_1Mul)sequential/LSTM_lay_0/lstm_cell/add_2:z:00sequential/LSTM_lay_0/lstm_cell/Const_4:output:0*
T0*(
_output_shapes
:??????????2'
%sequential/LSTM_lay_0/lstm_cell/Mul_1?
%sequential/LSTM_lay_0/lstm_cell/Add_3Add)sequential/LSTM_lay_0/lstm_cell/Mul_1:z:00sequential/LSTM_lay_0/lstm_cell/Const_5:output:0*
T0*(
_output_shapes
:??????????2'
%sequential/LSTM_lay_0/lstm_cell/Add_3?
9sequential/LSTM_lay_0/lstm_cell/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2;
9sequential/LSTM_lay_0/lstm_cell/clip_by_value_1/Minimum/y?
7sequential/LSTM_lay_0/lstm_cell/clip_by_value_1/MinimumMinimum)sequential/LSTM_lay_0/lstm_cell/Add_3:z:0Bsequential/LSTM_lay_0/lstm_cell/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:??????????29
7sequential/LSTM_lay_0/lstm_cell/clip_by_value_1/Minimum?
1sequential/LSTM_lay_0/lstm_cell/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    23
1sequential/LSTM_lay_0/lstm_cell/clip_by_value_1/y?
/sequential/LSTM_lay_0/lstm_cell/clip_by_value_1Maximum;sequential/LSTM_lay_0/lstm_cell/clip_by_value_1/Minimum:z:0:sequential/LSTM_lay_0/lstm_cell/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:??????????21
/sequential/LSTM_lay_0/lstm_cell/clip_by_value_1?
%sequential/LSTM_lay_0/lstm_cell/mul_2Mul3sequential/LSTM_lay_0/lstm_cell/clip_by_value_1:z:0&sequential/LSTM_lay_0/zeros_1:output:0*
T0*(
_output_shapes
:??????????2'
%sequential/LSTM_lay_0/lstm_cell/mul_2?
0sequential/LSTM_lay_0/lstm_cell/ReadVariableOp_2ReadVariableOp7sequential_lstm_lay_0_lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype022
0sequential/LSTM_lay_0/lstm_cell/ReadVariableOp_2?
5sequential/LSTM_lay_0/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       27
5sequential/LSTM_lay_0/lstm_cell/strided_slice_2/stack?
7sequential/LSTM_lay_0/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  29
7sequential/LSTM_lay_0/lstm_cell/strided_slice_2/stack_1?
7sequential/LSTM_lay_0/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7sequential/LSTM_lay_0/lstm_cell/strided_slice_2/stack_2?
/sequential/LSTM_lay_0/lstm_cell/strided_slice_2StridedSlice8sequential/LSTM_lay_0/lstm_cell/ReadVariableOp_2:value:0>sequential/LSTM_lay_0/lstm_cell/strided_slice_2/stack:output:0@sequential/LSTM_lay_0/lstm_cell/strided_slice_2/stack_1:output:0@sequential/LSTM_lay_0/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask21
/sequential/LSTM_lay_0/lstm_cell/strided_slice_2?
(sequential/LSTM_lay_0/lstm_cell/MatMul_6MatMul$sequential/LSTM_lay_0/zeros:output:08sequential/LSTM_lay_0/lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2*
(sequential/LSTM_lay_0/lstm_cell/MatMul_6?
%sequential/LSTM_lay_0/lstm_cell/add_4AddV22sequential/LSTM_lay_0/lstm_cell/BiasAdd_2:output:02sequential/LSTM_lay_0/lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2'
%sequential/LSTM_lay_0/lstm_cell/add_4?
'sequential/LSTM_lay_0/lstm_cell/SigmoidSigmoid)sequential/LSTM_lay_0/lstm_cell/add_4:z:0*
T0*(
_output_shapes
:??????????2)
'sequential/LSTM_lay_0/lstm_cell/Sigmoid?
%sequential/LSTM_lay_0/lstm_cell/mul_3Mul1sequential/LSTM_lay_0/lstm_cell/clip_by_value:z:0+sequential/LSTM_lay_0/lstm_cell/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2'
%sequential/LSTM_lay_0/lstm_cell/mul_3?
%sequential/LSTM_lay_0/lstm_cell/add_5AddV2)sequential/LSTM_lay_0/lstm_cell/mul_2:z:0)sequential/LSTM_lay_0/lstm_cell/mul_3:z:0*
T0*(
_output_shapes
:??????????2'
%sequential/LSTM_lay_0/lstm_cell/add_5?
0sequential/LSTM_lay_0/lstm_cell/ReadVariableOp_3ReadVariableOp7sequential_lstm_lay_0_lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype022
0sequential/LSTM_lay_0/lstm_cell/ReadVariableOp_3?
5sequential/LSTM_lay_0/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  27
5sequential/LSTM_lay_0/lstm_cell/strided_slice_3/stack?
7sequential/LSTM_lay_0/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        29
7sequential/LSTM_lay_0/lstm_cell/strided_slice_3/stack_1?
7sequential/LSTM_lay_0/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7sequential/LSTM_lay_0/lstm_cell/strided_slice_3/stack_2?
/sequential/LSTM_lay_0/lstm_cell/strided_slice_3StridedSlice8sequential/LSTM_lay_0/lstm_cell/ReadVariableOp_3:value:0>sequential/LSTM_lay_0/lstm_cell/strided_slice_3/stack:output:0@sequential/LSTM_lay_0/lstm_cell/strided_slice_3/stack_1:output:0@sequential/LSTM_lay_0/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask21
/sequential/LSTM_lay_0/lstm_cell/strided_slice_3?
(sequential/LSTM_lay_0/lstm_cell/MatMul_7MatMul$sequential/LSTM_lay_0/zeros:output:08sequential/LSTM_lay_0/lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2*
(sequential/LSTM_lay_0/lstm_cell/MatMul_7?
%sequential/LSTM_lay_0/lstm_cell/add_6AddV22sequential/LSTM_lay_0/lstm_cell/BiasAdd_3:output:02sequential/LSTM_lay_0/lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2'
%sequential/LSTM_lay_0/lstm_cell/add_6?
'sequential/LSTM_lay_0/lstm_cell/Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2)
'sequential/LSTM_lay_0/lstm_cell/Const_6?
'sequential/LSTM_lay_0/lstm_cell/Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2)
'sequential/LSTM_lay_0/lstm_cell/Const_7?
%sequential/LSTM_lay_0/lstm_cell/Mul_4Mul)sequential/LSTM_lay_0/lstm_cell/add_6:z:00sequential/LSTM_lay_0/lstm_cell/Const_6:output:0*
T0*(
_output_shapes
:??????????2'
%sequential/LSTM_lay_0/lstm_cell/Mul_4?
%sequential/LSTM_lay_0/lstm_cell/Add_7Add)sequential/LSTM_lay_0/lstm_cell/Mul_4:z:00sequential/LSTM_lay_0/lstm_cell/Const_7:output:0*
T0*(
_output_shapes
:??????????2'
%sequential/LSTM_lay_0/lstm_cell/Add_7?
9sequential/LSTM_lay_0/lstm_cell/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2;
9sequential/LSTM_lay_0/lstm_cell/clip_by_value_2/Minimum/y?
7sequential/LSTM_lay_0/lstm_cell/clip_by_value_2/MinimumMinimum)sequential/LSTM_lay_0/lstm_cell/Add_7:z:0Bsequential/LSTM_lay_0/lstm_cell/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:??????????29
7sequential/LSTM_lay_0/lstm_cell/clip_by_value_2/Minimum?
1sequential/LSTM_lay_0/lstm_cell/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    23
1sequential/LSTM_lay_0/lstm_cell/clip_by_value_2/y?
/sequential/LSTM_lay_0/lstm_cell/clip_by_value_2Maximum;sequential/LSTM_lay_0/lstm_cell/clip_by_value_2/Minimum:z:0:sequential/LSTM_lay_0/lstm_cell/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:??????????21
/sequential/LSTM_lay_0/lstm_cell/clip_by_value_2?
)sequential/LSTM_lay_0/lstm_cell/Sigmoid_1Sigmoid)sequential/LSTM_lay_0/lstm_cell/add_5:z:0*
T0*(
_output_shapes
:??????????2+
)sequential/LSTM_lay_0/lstm_cell/Sigmoid_1?
%sequential/LSTM_lay_0/lstm_cell/mul_5Mul3sequential/LSTM_lay_0/lstm_cell/clip_by_value_2:z:0-sequential/LSTM_lay_0/lstm_cell/Sigmoid_1:y:0*
T0*(
_output_shapes
:??????????2'
%sequential/LSTM_lay_0/lstm_cell/mul_5?
3sequential/LSTM_lay_0/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   25
3sequential/LSTM_lay_0/TensorArrayV2_1/element_shape?
%sequential/LSTM_lay_0/TensorArrayV2_1TensorListReserve<sequential/LSTM_lay_0/TensorArrayV2_1/element_shape:output:0.sequential/LSTM_lay_0/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02'
%sequential/LSTM_lay_0/TensorArrayV2_1z
sequential/LSTM_lay_0/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential/LSTM_lay_0/time?
.sequential/LSTM_lay_0/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????20
.sequential/LSTM_lay_0/while/maximum_iterations?
(sequential/LSTM_lay_0/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2*
(sequential/LSTM_lay_0/while/loop_counter?
sequential/LSTM_lay_0/whileWhile1sequential/LSTM_lay_0/while/loop_counter:output:07sequential/LSTM_lay_0/while/maximum_iterations:output:0#sequential/LSTM_lay_0/time:output:0.sequential/LSTM_lay_0/TensorArrayV2_1:handle:0$sequential/LSTM_lay_0/zeros:output:0&sequential/LSTM_lay_0/zeros_1:output:0.sequential/LSTM_lay_0/strided_slice_1:output:0Msequential/LSTM_lay_0/TensorArrayUnstack/TensorListFromTensor:output_handle:0=sequential_lstm_lay_0_lstm_cell_split_readvariableop_resource?sequential_lstm_lay_0_lstm_cell_split_1_readvariableop_resource7sequential_lstm_lay_0_lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*3
body+R)
'sequential_LSTM_lay_0_while_body_118524*3
cond+R)
'sequential_LSTM_lay_0_while_cond_118523*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
sequential/LSTM_lay_0/while?
Fsequential/LSTM_lay_0/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2H
Fsequential/LSTM_lay_0/TensorArrayV2Stack/TensorListStack/element_shape?
8sequential/LSTM_lay_0/TensorArrayV2Stack/TensorListStackTensorListStack$sequential/LSTM_lay_0/while:output:3Osequential/LSTM_lay_0/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype02:
8sequential/LSTM_lay_0/TensorArrayV2Stack/TensorListStack?
+sequential/LSTM_lay_0/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2-
+sequential/LSTM_lay_0/strided_slice_3/stack?
-sequential/LSTM_lay_0/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2/
-sequential/LSTM_lay_0/strided_slice_3/stack_1?
-sequential/LSTM_lay_0/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential/LSTM_lay_0/strided_slice_3/stack_2?
%sequential/LSTM_lay_0/strided_slice_3StridedSliceAsequential/LSTM_lay_0/TensorArrayV2Stack/TensorListStack:tensor:04sequential/LSTM_lay_0/strided_slice_3/stack:output:06sequential/LSTM_lay_0/strided_slice_3/stack_1:output:06sequential/LSTM_lay_0/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2'
%sequential/LSTM_lay_0/strided_slice_3?
&sequential/LSTM_lay_0/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2(
&sequential/LSTM_lay_0/transpose_1/perm?
!sequential/LSTM_lay_0/transpose_1	TransposeAsequential/LSTM_lay_0/TensorArrayV2Stack/TensorListStack:tensor:0/sequential/LSTM_lay_0/transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2#
!sequential/LSTM_lay_0/transpose_1?
!sequential/Dropout_lay_0/IdentityIdentity.sequential/LSTM_lay_0/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2#
!sequential/Dropout_lay_0/Identity?
,sequential/Dense_lay_1/MatMul/ReadVariableOpReadVariableOp5sequential_dense_lay_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02.
,sequential/Dense_lay_1/MatMul/ReadVariableOp?
sequential/Dense_lay_1/MatMulMatMul*sequential/Dropout_lay_0/Identity:output:04sequential/Dense_lay_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential/Dense_lay_1/MatMul?
-sequential/Dense_lay_1/BiasAdd/ReadVariableOpReadVariableOp6sequential_dense_lay_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-sequential/Dense_lay_1/BiasAdd/ReadVariableOp?
sequential/Dense_lay_1/BiasAddBiasAdd'sequential/Dense_lay_1/MatMul:product:05sequential/Dense_lay_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
sequential/Dense_lay_1/BiasAdd?
sequential/Dense_lay_1/ReluRelu'sequential/Dense_lay_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
sequential/Dense_lay_1/Relu?
+sequential/Output_lay/MatMul/ReadVariableOpReadVariableOp4sequential_output_lay_matmul_readvariableop_resource*
_output_shapes
:	?`*
dtype02-
+sequential/Output_lay/MatMul/ReadVariableOp?
sequential/Output_lay/MatMulMatMul)sequential/Dense_lay_1/Relu:activations:03sequential/Output_lay/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????`2
sequential/Output_lay/MatMul?
,sequential/Output_lay/BiasAdd/ReadVariableOpReadVariableOp5sequential_output_lay_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype02.
,sequential/Output_lay/BiasAdd/ReadVariableOp?
sequential/Output_lay/BiasAddBiasAdd&sequential/Output_lay/MatMul:product:04sequential/Output_lay/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????`2
sequential/Output_lay/BiasAdd?
IdentityIdentity&sequential/Output_lay/BiasAdd:output:0.^sequential/Dense_lay_1/BiasAdd/ReadVariableOp-^sequential/Dense_lay_1/MatMul/ReadVariableOp/^sequential/LSTM_lay_0/lstm_cell/ReadVariableOp1^sequential/LSTM_lay_0/lstm_cell/ReadVariableOp_11^sequential/LSTM_lay_0/lstm_cell/ReadVariableOp_21^sequential/LSTM_lay_0/lstm_cell/ReadVariableOp_35^sequential/LSTM_lay_0/lstm_cell/split/ReadVariableOp7^sequential/LSTM_lay_0/lstm_cell/split_1/ReadVariableOp^sequential/LSTM_lay_0/while-^sequential/Output_lay/BiasAdd/ReadVariableOp,^sequential/Output_lay/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????`2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:??????????:::::::2^
-sequential/Dense_lay_1/BiasAdd/ReadVariableOp-sequential/Dense_lay_1/BiasAdd/ReadVariableOp2\
,sequential/Dense_lay_1/MatMul/ReadVariableOp,sequential/Dense_lay_1/MatMul/ReadVariableOp2`
.sequential/LSTM_lay_0/lstm_cell/ReadVariableOp.sequential/LSTM_lay_0/lstm_cell/ReadVariableOp2d
0sequential/LSTM_lay_0/lstm_cell/ReadVariableOp_10sequential/LSTM_lay_0/lstm_cell/ReadVariableOp_12d
0sequential/LSTM_lay_0/lstm_cell/ReadVariableOp_20sequential/LSTM_lay_0/lstm_cell/ReadVariableOp_22d
0sequential/LSTM_lay_0/lstm_cell/ReadVariableOp_30sequential/LSTM_lay_0/lstm_cell/ReadVariableOp_32l
4sequential/LSTM_lay_0/lstm_cell/split/ReadVariableOp4sequential/LSTM_lay_0/lstm_cell/split/ReadVariableOp2p
6sequential/LSTM_lay_0/lstm_cell/split_1/ReadVariableOp6sequential/LSTM_lay_0/lstm_cell/split_1/ReadVariableOp2:
sequential/LSTM_lay_0/whilesequential/LSTM_lay_0/while2\
,sequential/Output_lay/BiasAdd/ReadVariableOp,sequential/Output_lay/BiasAdd/ReadVariableOp2Z
+sequential/Output_lay/MatMul/ReadVariableOp+sequential/Output_lay/MatMul/ReadVariableOp:^ Z
,
_output_shapes
:??????????
*
_user_specified_nameLSTM_lay_0_input
?
?
while_cond_120923
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_120923___redundant_placeholder04
0while_while_cond_120923___redundant_placeholder14
0while_while_cond_120923___redundant_placeholder24
0while_while_cond_120923___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
F__inference_sequential_layer_call_and_return_conditional_losses_120143

inputs
lstm_lay_0_120124
lstm_lay_0_120126
lstm_lay_0_120128
dense_lay_1_120132
dense_lay_1_120134
output_lay_120137
output_lay_120139
identity??#Dense_lay_1/StatefulPartitionedCall?"LSTM_lay_0/StatefulPartitionedCall?"Output_lay/StatefulPartitionedCall?
"LSTM_lay_0/StatefulPartitionedCallStatefulPartitionedCallinputslstm_lay_0_120124lstm_lay_0_120126lstm_lay_0_120128*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_LSTM_lay_0_layer_call_and_return_conditional_losses_1199412$
"LSTM_lay_0/StatefulPartitionedCall?
Dropout_lay_0/PartitionedCallPartitionedCall+LSTM_lay_0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_Dropout_lay_0_layer_call_and_return_conditional_losses_1199882
Dropout_lay_0/PartitionedCall?
#Dense_lay_1/StatefulPartitionedCallStatefulPartitionedCall&Dropout_lay_0/PartitionedCall:output:0dense_lay_1_120132dense_lay_1_120134*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_Dense_lay_1_layer_call_and_return_conditional_losses_1200122%
#Dense_lay_1/StatefulPartitionedCall?
"Output_lay/StatefulPartitionedCallStatefulPartitionedCall,Dense_lay_1/StatefulPartitionedCall:output:0output_lay_120137output_lay_120139*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????`*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_Output_lay_layer_call_and_return_conditional_losses_1200382$
"Output_lay/StatefulPartitionedCall?
IdentityIdentity+Output_lay/StatefulPartitionedCall:output:0$^Dense_lay_1/StatefulPartitionedCall#^LSTM_lay_0/StatefulPartitionedCall#^Output_lay/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????`2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:??????????:::::::2J
#Dense_lay_1/StatefulPartitionedCall#Dense_lay_1/StatefulPartitionedCall2H
"LSTM_lay_0/StatefulPartitionedCall"LSTM_lay_0/StatefulPartitionedCall2H
"Output_lay/StatefulPartitionedCall"Output_lay/StatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
while_cond_121191
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_121191___redundant_placeholder04
0while_while_cond_121191___redundant_placeholder14
0while_while_cond_121191___redundant_placeholder24
0while_while_cond_121191___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
F__inference_sequential_layer_call_and_return_conditional_losses_120055
lstm_lay_0_input
lstm_lay_0_119964
lstm_lay_0_119966
lstm_lay_0_119968
dense_lay_1_120023
dense_lay_1_120025
output_lay_120049
output_lay_120051
identity??#Dense_lay_1/StatefulPartitionedCall?%Dropout_lay_0/StatefulPartitionedCall?"LSTM_lay_0/StatefulPartitionedCall?"Output_lay/StatefulPartitionedCall?
"LSTM_lay_0/StatefulPartitionedCallStatefulPartitionedCalllstm_lay_0_inputlstm_lay_0_119964lstm_lay_0_119966lstm_lay_0_119968*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_LSTM_lay_0_layer_call_and_return_conditional_losses_1196732$
"LSTM_lay_0/StatefulPartitionedCall?
%Dropout_lay_0/StatefulPartitionedCallStatefulPartitionedCall+LSTM_lay_0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_Dropout_lay_0_layer_call_and_return_conditional_losses_1199832'
%Dropout_lay_0/StatefulPartitionedCall?
#Dense_lay_1/StatefulPartitionedCallStatefulPartitionedCall.Dropout_lay_0/StatefulPartitionedCall:output:0dense_lay_1_120023dense_lay_1_120025*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_Dense_lay_1_layer_call_and_return_conditional_losses_1200122%
#Dense_lay_1/StatefulPartitionedCall?
"Output_lay/StatefulPartitionedCallStatefulPartitionedCall,Dense_lay_1/StatefulPartitionedCall:output:0output_lay_120049output_lay_120051*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????`*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_Output_lay_layer_call_and_return_conditional_losses_1200382$
"Output_lay/StatefulPartitionedCall?
IdentityIdentity+Output_lay/StatefulPartitionedCall:output:0$^Dense_lay_1/StatefulPartitionedCall&^Dropout_lay_0/StatefulPartitionedCall#^LSTM_lay_0/StatefulPartitionedCall#^Output_lay/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????`2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:??????????:::::::2J
#Dense_lay_1/StatefulPartitionedCall#Dense_lay_1/StatefulPartitionedCall2N
%Dropout_lay_0/StatefulPartitionedCall%Dropout_lay_0/StatefulPartitionedCall2H
"LSTM_lay_0/StatefulPartitionedCall"LSTM_lay_0/StatefulPartitionedCall2H
"Output_lay/StatefulPartitionedCall"Output_lay/StatefulPartitionedCall:^ Z
,
_output_shapes
:??????????
*
_user_specified_nameLSTM_lay_0_input
??
?
while_body_119531
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_03
/while_lstm_cell_split_readvariableop_resource_05
1while_lstm_cell_split_1_readvariableop_resource_0-
)while_lstm_cell_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor1
-while_lstm_cell_split_readvariableop_resource3
/while_lstm_cell_split_1_readvariableop_resource+
'while_lstm_cell_readvariableop_resource??while/lstm_cell/ReadVariableOp? while/lstm_cell/ReadVariableOp_1? while/lstm_cell/ReadVariableOp_2? while/lstm_cell/ReadVariableOp_3?$while/lstm_cell/split/ReadVariableOp?&while/lstm_cell/split_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemp
while/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell/Const?
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2!
while/lstm_cell/split/split_dim?
$while/lstm_cell/split/ReadVariableOpReadVariableOp/while_lstm_cell_split_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02&
$while/lstm_cell/split/ReadVariableOp?
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0,while/lstm_cell/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
??:
??:
??:
??*
	num_split2
while/lstm_cell/split?
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/MatMul?
while/lstm_cell/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:1*
T0*(
_output_shapes
:??????????2
while/lstm_cell/MatMul_1?
while/lstm_cell/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:2*
T0*(
_output_shapes
:??????????2
while/lstm_cell/MatMul_2?
while/lstm_cell/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:3*
T0*(
_output_shapes
:??????????2
while/lstm_cell/MatMul_3t
while/lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell/Const_1?
!while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!while/lstm_cell/split_1/split_dim?
&while/lstm_cell/split_1/ReadVariableOpReadVariableOp1while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
:?*
dtype02(
&while/lstm_cell/split_1/ReadVariableOp?
while/lstm_cell/split_1Split*while/lstm_cell/split_1/split_dim:output:0.while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2
while/lstm_cell/split_1?
while/lstm_cell/BiasAddBiasAdd while/lstm_cell/MatMul:product:0 while/lstm_cell/split_1:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/BiasAdd?
while/lstm_cell/BiasAdd_1BiasAdd"while/lstm_cell/MatMul_1:product:0 while/lstm_cell/split_1:output:1*
T0*(
_output_shapes
:??????????2
while/lstm_cell/BiasAdd_1?
while/lstm_cell/BiasAdd_2BiasAdd"while/lstm_cell/MatMul_2:product:0 while/lstm_cell/split_1:output:2*
T0*(
_output_shapes
:??????????2
while/lstm_cell/BiasAdd_2?
while/lstm_cell/BiasAdd_3BiasAdd"while/lstm_cell/MatMul_3:product:0 while/lstm_cell/split_1:output:3*
T0*(
_output_shapes
:??????????2
while/lstm_cell/BiasAdd_3?
while/lstm_cell/ReadVariableOpReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02 
while/lstm_cell/ReadVariableOp?
#while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2%
#while/lstm_cell/strided_slice/stack?
%while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2'
%while/lstm_cell/strided_slice/stack_1?
%while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2'
%while/lstm_cell/strided_slice/stack_2?
while/lstm_cell/strided_sliceStridedSlice&while/lstm_cell/ReadVariableOp:value:0,while/lstm_cell/strided_slice/stack:output:0.while/lstm_cell/strided_slice/stack_1:output:0.while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
while/lstm_cell/strided_slice?
while/lstm_cell/MatMul_4MatMulwhile_placeholder_2&while/lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/MatMul_4?
while/lstm_cell/addAddV2 while/lstm_cell/BiasAdd:output:0"while/lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/addw
while/lstm_cell/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/lstm_cell/Const_2w
while/lstm_cell/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/lstm_cell/Const_3?
while/lstm_cell/MulMulwhile/lstm_cell/add:z:0 while/lstm_cell/Const_2:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/Mul?
while/lstm_cell/Add_1Addwhile/lstm_cell/Mul:z:0 while/lstm_cell/Const_3:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/Add_1?
'while/lstm_cell/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2)
'while/lstm_cell/clip_by_value/Minimum/y?
%while/lstm_cell/clip_by_value/MinimumMinimumwhile/lstm_cell/Add_1:z:00while/lstm_cell/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:??????????2'
%while/lstm_cell/clip_by_value/Minimum?
while/lstm_cell/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
while/lstm_cell/clip_by_value/y?
while/lstm_cell/clip_by_valueMaximum)while/lstm_cell/clip_by_value/Minimum:z:0(while/lstm_cell/clip_by_value/y:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/clip_by_value?
 while/lstm_cell/ReadVariableOp_1ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02"
 while/lstm_cell/ReadVariableOp_1?
%while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2'
%while/lstm_cell/strided_slice_1/stack?
'while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2)
'while/lstm_cell/strided_slice_1/stack_1?
'while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_1/stack_2?
while/lstm_cell/strided_slice_1StridedSlice(while/lstm_cell/ReadVariableOp_1:value:0.while/lstm_cell/strided_slice_1/stack:output:00while/lstm_cell/strided_slice_1/stack_1:output:00while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2!
while/lstm_cell/strided_slice_1?
while/lstm_cell/MatMul_5MatMulwhile_placeholder_2(while/lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/MatMul_5?
while/lstm_cell/add_2AddV2"while/lstm_cell/BiasAdd_1:output:0"while/lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/add_2w
while/lstm_cell/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/lstm_cell/Const_4w
while/lstm_cell/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/lstm_cell/Const_5?
while/lstm_cell/Mul_1Mulwhile/lstm_cell/add_2:z:0 while/lstm_cell/Const_4:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/Mul_1?
while/lstm_cell/Add_3Addwhile/lstm_cell/Mul_1:z:0 while/lstm_cell/Const_5:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/Add_3?
)while/lstm_cell/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2+
)while/lstm_cell/clip_by_value_1/Minimum/y?
'while/lstm_cell/clip_by_value_1/MinimumMinimumwhile/lstm_cell/Add_3:z:02while/lstm_cell/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:??????????2)
'while/lstm_cell/clip_by_value_1/Minimum?
!while/lstm_cell/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!while/lstm_cell/clip_by_value_1/y?
while/lstm_cell/clip_by_value_1Maximum+while/lstm_cell/clip_by_value_1/Minimum:z:0*while/lstm_cell/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:??????????2!
while/lstm_cell/clip_by_value_1?
while/lstm_cell/mul_2Mul#while/lstm_cell/clip_by_value_1:z:0while_placeholder_3*
T0*(
_output_shapes
:??????????2
while/lstm_cell/mul_2?
 while/lstm_cell/ReadVariableOp_2ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02"
 while/lstm_cell/ReadVariableOp_2?
%while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2'
%while/lstm_cell/strided_slice_2/stack?
'while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2)
'while/lstm_cell/strided_slice_2/stack_1?
'while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_2/stack_2?
while/lstm_cell/strided_slice_2StridedSlice(while/lstm_cell/ReadVariableOp_2:value:0.while/lstm_cell/strided_slice_2/stack:output:00while/lstm_cell/strided_slice_2/stack_1:output:00while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2!
while/lstm_cell/strided_slice_2?
while/lstm_cell/MatMul_6MatMulwhile_placeholder_2(while/lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/MatMul_6?
while/lstm_cell/add_4AddV2"while/lstm_cell/BiasAdd_2:output:0"while/lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/add_4?
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/add_4:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/Sigmoid?
while/lstm_cell/mul_3Mul!while/lstm_cell/clip_by_value:z:0while/lstm_cell/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/mul_3?
while/lstm_cell/add_5AddV2while/lstm_cell/mul_2:z:0while/lstm_cell/mul_3:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/add_5?
 while/lstm_cell/ReadVariableOp_3ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02"
 while/lstm_cell/ReadVariableOp_3?
%while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2'
%while/lstm_cell/strided_slice_3/stack?
'while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'while/lstm_cell/strided_slice_3/stack_1?
'while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_3/stack_2?
while/lstm_cell/strided_slice_3StridedSlice(while/lstm_cell/ReadVariableOp_3:value:0.while/lstm_cell/strided_slice_3/stack:output:00while/lstm_cell/strided_slice_3/stack_1:output:00while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2!
while/lstm_cell/strided_slice_3?
while/lstm_cell/MatMul_7MatMulwhile_placeholder_2(while/lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/MatMul_7?
while/lstm_cell/add_6AddV2"while/lstm_cell/BiasAdd_3:output:0"while/lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/add_6w
while/lstm_cell/Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/lstm_cell/Const_6w
while/lstm_cell/Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/lstm_cell/Const_7?
while/lstm_cell/Mul_4Mulwhile/lstm_cell/add_6:z:0 while/lstm_cell/Const_6:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/Mul_4?
while/lstm_cell/Add_7Addwhile/lstm_cell/Mul_4:z:0 while/lstm_cell/Const_7:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/Add_7?
)while/lstm_cell/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2+
)while/lstm_cell/clip_by_value_2/Minimum/y?
'while/lstm_cell/clip_by_value_2/MinimumMinimumwhile/lstm_cell/Add_7:z:02while/lstm_cell/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:??????????2)
'while/lstm_cell/clip_by_value_2/Minimum?
!while/lstm_cell/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!while/lstm_cell/clip_by_value_2/y?
while/lstm_cell/clip_by_value_2Maximum+while/lstm_cell/clip_by_value_2/Minimum:z:0*while/lstm_cell/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:??????????2!
while/lstm_cell/clip_by_value_2?
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/add_5:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/Sigmoid_1?
while/lstm_cell/mul_5Mul#while/lstm_cell/clip_by_value_2:z:0while/lstm_cell/Sigmoid_1:y:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/mul_5?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_5:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:0^while/lstm_cell/ReadVariableOp!^while/lstm_cell/ReadVariableOp_1!^while/lstm_cell/ReadVariableOp_2!^while/lstm_cell/ReadVariableOp_3%^while/lstm_cell/split/ReadVariableOp'^while/lstm_cell/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations^while/lstm_cell/ReadVariableOp!^while/lstm_cell/ReadVariableOp_1!^while/lstm_cell/ReadVariableOp_2!^while/lstm_cell/ReadVariableOp_3%^while/lstm_cell/split/ReadVariableOp'^while/lstm_cell/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0^while/lstm_cell/ReadVariableOp!^while/lstm_cell/ReadVariableOp_1!^while/lstm_cell/ReadVariableOp_2!^while/lstm_cell/ReadVariableOp_3%^while/lstm_cell/split/ReadVariableOp'^while/lstm_cell/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/lstm_cell/ReadVariableOp!^while/lstm_cell/ReadVariableOp_1!^while/lstm_cell/ReadVariableOp_2!^while/lstm_cell/ReadVariableOp_3%^while/lstm_cell/split/ReadVariableOp'^while/lstm_cell/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell/mul_5:z:0^while/lstm_cell/ReadVariableOp!^while/lstm_cell/ReadVariableOp_1!^while/lstm_cell/ReadVariableOp_2!^while/lstm_cell/ReadVariableOp_3%^while/lstm_cell/split/ReadVariableOp'^while/lstm_cell/split_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell/add_5:z:0^while/lstm_cell/ReadVariableOp!^while/lstm_cell/ReadVariableOp_1!^while/lstm_cell/ReadVariableOp_2!^while/lstm_cell/ReadVariableOp_3%^while/lstm_cell/split/ReadVariableOp'^while/lstm_cell/split_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"T
'while_lstm_cell_readvariableop_resource)while_lstm_cell_readvariableop_resource_0"d
/while_lstm_cell_split_1_readvariableop_resource1while_lstm_cell_split_1_readvariableop_resource_0"`
-while_lstm_cell_split_readvariableop_resource/while_lstm_cell_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :??????????:??????????: : :::2@
while/lstm_cell/ReadVariableOpwhile/lstm_cell/ReadVariableOp2D
 while/lstm_cell/ReadVariableOp_1 while/lstm_cell/ReadVariableOp_12D
 while/lstm_cell/ReadVariableOp_2 while/lstm_cell/ReadVariableOp_22D
 while/lstm_cell/ReadVariableOp_3 while/lstm_cell/ReadVariableOp_32L
$while/lstm_cell/split/ReadVariableOp$while/lstm_cell/split/ReadVariableOp2P
&while/lstm_cell/split_1/ReadVariableOp&while/lstm_cell/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
??
?
F__inference_sequential_layer_call_and_return_conditional_losses_120760

inputs6
2lstm_lay_0_lstm_cell_split_readvariableop_resource8
4lstm_lay_0_lstm_cell_split_1_readvariableop_resource0
,lstm_lay_0_lstm_cell_readvariableop_resource.
*dense_lay_1_matmul_readvariableop_resource/
+dense_lay_1_biasadd_readvariableop_resource-
)output_lay_matmul_readvariableop_resource.
*output_lay_biasadd_readvariableop_resource
identity??"Dense_lay_1/BiasAdd/ReadVariableOp?!Dense_lay_1/MatMul/ReadVariableOp?#LSTM_lay_0/lstm_cell/ReadVariableOp?%LSTM_lay_0/lstm_cell/ReadVariableOp_1?%LSTM_lay_0/lstm_cell/ReadVariableOp_2?%LSTM_lay_0/lstm_cell/ReadVariableOp_3?)LSTM_lay_0/lstm_cell/split/ReadVariableOp?+LSTM_lay_0/lstm_cell/split_1/ReadVariableOp?LSTM_lay_0/while?!Output_lay/BiasAdd/ReadVariableOp? Output_lay/MatMul/ReadVariableOpZ
LSTM_lay_0/ShapeShapeinputs*
T0*
_output_shapes
:2
LSTM_lay_0/Shape?
LSTM_lay_0/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
LSTM_lay_0/strided_slice/stack?
 LSTM_lay_0/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 LSTM_lay_0/strided_slice/stack_1?
 LSTM_lay_0/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 LSTM_lay_0/strided_slice/stack_2?
LSTM_lay_0/strided_sliceStridedSliceLSTM_lay_0/Shape:output:0'LSTM_lay_0/strided_slice/stack:output:0)LSTM_lay_0/strided_slice/stack_1:output:0)LSTM_lay_0/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
LSTM_lay_0/strided_slices
LSTM_lay_0/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
LSTM_lay_0/zeros/mul/y?
LSTM_lay_0/zeros/mulMul!LSTM_lay_0/strided_slice:output:0LSTM_lay_0/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
LSTM_lay_0/zeros/mulu
LSTM_lay_0/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
LSTM_lay_0/zeros/Less/y?
LSTM_lay_0/zeros/LessLessLSTM_lay_0/zeros/mul:z:0 LSTM_lay_0/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
LSTM_lay_0/zeros/Lessy
LSTM_lay_0/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
LSTM_lay_0/zeros/packed/1?
LSTM_lay_0/zeros/packedPack!LSTM_lay_0/strided_slice:output:0"LSTM_lay_0/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
LSTM_lay_0/zeros/packedu
LSTM_lay_0/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
LSTM_lay_0/zeros/Const?
LSTM_lay_0/zerosFill LSTM_lay_0/zeros/packed:output:0LSTM_lay_0/zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
LSTM_lay_0/zerosw
LSTM_lay_0/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
LSTM_lay_0/zeros_1/mul/y?
LSTM_lay_0/zeros_1/mulMul!LSTM_lay_0/strided_slice:output:0!LSTM_lay_0/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
LSTM_lay_0/zeros_1/muly
LSTM_lay_0/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
LSTM_lay_0/zeros_1/Less/y?
LSTM_lay_0/zeros_1/LessLessLSTM_lay_0/zeros_1/mul:z:0"LSTM_lay_0/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
LSTM_lay_0/zeros_1/Less}
LSTM_lay_0/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
LSTM_lay_0/zeros_1/packed/1?
LSTM_lay_0/zeros_1/packedPack!LSTM_lay_0/strided_slice:output:0$LSTM_lay_0/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
LSTM_lay_0/zeros_1/packedy
LSTM_lay_0/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
LSTM_lay_0/zeros_1/Const?
LSTM_lay_0/zeros_1Fill"LSTM_lay_0/zeros_1/packed:output:0!LSTM_lay_0/zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2
LSTM_lay_0/zeros_1?
LSTM_lay_0/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
LSTM_lay_0/transpose/perm?
LSTM_lay_0/transpose	Transposeinputs"LSTM_lay_0/transpose/perm:output:0*
T0*,
_output_shapes
:??????????2
LSTM_lay_0/transposep
LSTM_lay_0/Shape_1ShapeLSTM_lay_0/transpose:y:0*
T0*
_output_shapes
:2
LSTM_lay_0/Shape_1?
 LSTM_lay_0/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 LSTM_lay_0/strided_slice_1/stack?
"LSTM_lay_0/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"LSTM_lay_0/strided_slice_1/stack_1?
"LSTM_lay_0/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"LSTM_lay_0/strided_slice_1/stack_2?
LSTM_lay_0/strided_slice_1StridedSliceLSTM_lay_0/Shape_1:output:0)LSTM_lay_0/strided_slice_1/stack:output:0+LSTM_lay_0/strided_slice_1/stack_1:output:0+LSTM_lay_0/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
LSTM_lay_0/strided_slice_1?
&LSTM_lay_0/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2(
&LSTM_lay_0/TensorArrayV2/element_shape?
LSTM_lay_0/TensorArrayV2TensorListReserve/LSTM_lay_0/TensorArrayV2/element_shape:output:0#LSTM_lay_0/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
LSTM_lay_0/TensorArrayV2?
@LSTM_lay_0/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2B
@LSTM_lay_0/TensorArrayUnstack/TensorListFromTensor/element_shape?
2LSTM_lay_0/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorLSTM_lay_0/transpose:y:0ILSTM_lay_0/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type024
2LSTM_lay_0/TensorArrayUnstack/TensorListFromTensor?
 LSTM_lay_0/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 LSTM_lay_0/strided_slice_2/stack?
"LSTM_lay_0/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"LSTM_lay_0/strided_slice_2/stack_1?
"LSTM_lay_0/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"LSTM_lay_0/strided_slice_2/stack_2?
LSTM_lay_0/strided_slice_2StridedSliceLSTM_lay_0/transpose:y:0)LSTM_lay_0/strided_slice_2/stack:output:0+LSTM_lay_0/strided_slice_2/stack_1:output:0+LSTM_lay_0/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
LSTM_lay_0/strided_slice_2z
LSTM_lay_0/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
LSTM_lay_0/lstm_cell/Const?
$LSTM_lay_0/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$LSTM_lay_0/lstm_cell/split/split_dim?
)LSTM_lay_0/lstm_cell/split/ReadVariableOpReadVariableOp2lstm_lay_0_lstm_cell_split_readvariableop_resource* 
_output_shapes
:
??*
dtype02+
)LSTM_lay_0/lstm_cell/split/ReadVariableOp?
LSTM_lay_0/lstm_cell/splitSplit-LSTM_lay_0/lstm_cell/split/split_dim:output:01LSTM_lay_0/lstm_cell/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
??:
??:
??:
??*
	num_split2
LSTM_lay_0/lstm_cell/split?
LSTM_lay_0/lstm_cell/MatMulMatMul#LSTM_lay_0/strided_slice_2:output:0#LSTM_lay_0/lstm_cell/split:output:0*
T0*(
_output_shapes
:??????????2
LSTM_lay_0/lstm_cell/MatMul?
LSTM_lay_0/lstm_cell/MatMul_1MatMul#LSTM_lay_0/strided_slice_2:output:0#LSTM_lay_0/lstm_cell/split:output:1*
T0*(
_output_shapes
:??????????2
LSTM_lay_0/lstm_cell/MatMul_1?
LSTM_lay_0/lstm_cell/MatMul_2MatMul#LSTM_lay_0/strided_slice_2:output:0#LSTM_lay_0/lstm_cell/split:output:2*
T0*(
_output_shapes
:??????????2
LSTM_lay_0/lstm_cell/MatMul_2?
LSTM_lay_0/lstm_cell/MatMul_3MatMul#LSTM_lay_0/strided_slice_2:output:0#LSTM_lay_0/lstm_cell/split:output:3*
T0*(
_output_shapes
:??????????2
LSTM_lay_0/lstm_cell/MatMul_3~
LSTM_lay_0/lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
LSTM_lay_0/lstm_cell/Const_1?
&LSTM_lay_0/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2(
&LSTM_lay_0/lstm_cell/split_1/split_dim?
+LSTM_lay_0/lstm_cell/split_1/ReadVariableOpReadVariableOp4lstm_lay_0_lstm_cell_split_1_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+LSTM_lay_0/lstm_cell/split_1/ReadVariableOp?
LSTM_lay_0/lstm_cell/split_1Split/LSTM_lay_0/lstm_cell/split_1/split_dim:output:03LSTM_lay_0/lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2
LSTM_lay_0/lstm_cell/split_1?
LSTM_lay_0/lstm_cell/BiasAddBiasAdd%LSTM_lay_0/lstm_cell/MatMul:product:0%LSTM_lay_0/lstm_cell/split_1:output:0*
T0*(
_output_shapes
:??????????2
LSTM_lay_0/lstm_cell/BiasAdd?
LSTM_lay_0/lstm_cell/BiasAdd_1BiasAdd'LSTM_lay_0/lstm_cell/MatMul_1:product:0%LSTM_lay_0/lstm_cell/split_1:output:1*
T0*(
_output_shapes
:??????????2 
LSTM_lay_0/lstm_cell/BiasAdd_1?
LSTM_lay_0/lstm_cell/BiasAdd_2BiasAdd'LSTM_lay_0/lstm_cell/MatMul_2:product:0%LSTM_lay_0/lstm_cell/split_1:output:2*
T0*(
_output_shapes
:??????????2 
LSTM_lay_0/lstm_cell/BiasAdd_2?
LSTM_lay_0/lstm_cell/BiasAdd_3BiasAdd'LSTM_lay_0/lstm_cell/MatMul_3:product:0%LSTM_lay_0/lstm_cell/split_1:output:3*
T0*(
_output_shapes
:??????????2 
LSTM_lay_0/lstm_cell/BiasAdd_3?
#LSTM_lay_0/lstm_cell/ReadVariableOpReadVariableOp,lstm_lay_0_lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02%
#LSTM_lay_0/lstm_cell/ReadVariableOp?
(LSTM_lay_0/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2*
(LSTM_lay_0/lstm_cell/strided_slice/stack?
*LSTM_lay_0/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2,
*LSTM_lay_0/lstm_cell/strided_slice/stack_1?
*LSTM_lay_0/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*LSTM_lay_0/lstm_cell/strided_slice/stack_2?
"LSTM_lay_0/lstm_cell/strided_sliceStridedSlice+LSTM_lay_0/lstm_cell/ReadVariableOp:value:01LSTM_lay_0/lstm_cell/strided_slice/stack:output:03LSTM_lay_0/lstm_cell/strided_slice/stack_1:output:03LSTM_lay_0/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2$
"LSTM_lay_0/lstm_cell/strided_slice?
LSTM_lay_0/lstm_cell/MatMul_4MatMulLSTM_lay_0/zeros:output:0+LSTM_lay_0/lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
LSTM_lay_0/lstm_cell/MatMul_4?
LSTM_lay_0/lstm_cell/addAddV2%LSTM_lay_0/lstm_cell/BiasAdd:output:0'LSTM_lay_0/lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
LSTM_lay_0/lstm_cell/add?
LSTM_lay_0/lstm_cell/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
LSTM_lay_0/lstm_cell/Const_2?
LSTM_lay_0/lstm_cell/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
LSTM_lay_0/lstm_cell/Const_3?
LSTM_lay_0/lstm_cell/MulMulLSTM_lay_0/lstm_cell/add:z:0%LSTM_lay_0/lstm_cell/Const_2:output:0*
T0*(
_output_shapes
:??????????2
LSTM_lay_0/lstm_cell/Mul?
LSTM_lay_0/lstm_cell/Add_1AddLSTM_lay_0/lstm_cell/Mul:z:0%LSTM_lay_0/lstm_cell/Const_3:output:0*
T0*(
_output_shapes
:??????????2
LSTM_lay_0/lstm_cell/Add_1?
,LSTM_lay_0/lstm_cell/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2.
,LSTM_lay_0/lstm_cell/clip_by_value/Minimum/y?
*LSTM_lay_0/lstm_cell/clip_by_value/MinimumMinimumLSTM_lay_0/lstm_cell/Add_1:z:05LSTM_lay_0/lstm_cell/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:??????????2,
*LSTM_lay_0/lstm_cell/clip_by_value/Minimum?
$LSTM_lay_0/lstm_cell/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2&
$LSTM_lay_0/lstm_cell/clip_by_value/y?
"LSTM_lay_0/lstm_cell/clip_by_valueMaximum.LSTM_lay_0/lstm_cell/clip_by_value/Minimum:z:0-LSTM_lay_0/lstm_cell/clip_by_value/y:output:0*
T0*(
_output_shapes
:??????????2$
"LSTM_lay_0/lstm_cell/clip_by_value?
%LSTM_lay_0/lstm_cell/ReadVariableOp_1ReadVariableOp,lstm_lay_0_lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02'
%LSTM_lay_0/lstm_cell/ReadVariableOp_1?
*LSTM_lay_0/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2,
*LSTM_lay_0/lstm_cell/strided_slice_1/stack?
,LSTM_lay_0/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2.
,LSTM_lay_0/lstm_cell/strided_slice_1/stack_1?
,LSTM_lay_0/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2.
,LSTM_lay_0/lstm_cell/strided_slice_1/stack_2?
$LSTM_lay_0/lstm_cell/strided_slice_1StridedSlice-LSTM_lay_0/lstm_cell/ReadVariableOp_1:value:03LSTM_lay_0/lstm_cell/strided_slice_1/stack:output:05LSTM_lay_0/lstm_cell/strided_slice_1/stack_1:output:05LSTM_lay_0/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2&
$LSTM_lay_0/lstm_cell/strided_slice_1?
LSTM_lay_0/lstm_cell/MatMul_5MatMulLSTM_lay_0/zeros:output:0-LSTM_lay_0/lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
LSTM_lay_0/lstm_cell/MatMul_5?
LSTM_lay_0/lstm_cell/add_2AddV2'LSTM_lay_0/lstm_cell/BiasAdd_1:output:0'LSTM_lay_0/lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
LSTM_lay_0/lstm_cell/add_2?
LSTM_lay_0/lstm_cell/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
LSTM_lay_0/lstm_cell/Const_4?
LSTM_lay_0/lstm_cell/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
LSTM_lay_0/lstm_cell/Const_5?
LSTM_lay_0/lstm_cell/Mul_1MulLSTM_lay_0/lstm_cell/add_2:z:0%LSTM_lay_0/lstm_cell/Const_4:output:0*
T0*(
_output_shapes
:??????????2
LSTM_lay_0/lstm_cell/Mul_1?
LSTM_lay_0/lstm_cell/Add_3AddLSTM_lay_0/lstm_cell/Mul_1:z:0%LSTM_lay_0/lstm_cell/Const_5:output:0*
T0*(
_output_shapes
:??????????2
LSTM_lay_0/lstm_cell/Add_3?
.LSTM_lay_0/lstm_cell/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??20
.LSTM_lay_0/lstm_cell/clip_by_value_1/Minimum/y?
,LSTM_lay_0/lstm_cell/clip_by_value_1/MinimumMinimumLSTM_lay_0/lstm_cell/Add_3:z:07LSTM_lay_0/lstm_cell/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:??????????2.
,LSTM_lay_0/lstm_cell/clip_by_value_1/Minimum?
&LSTM_lay_0/lstm_cell/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2(
&LSTM_lay_0/lstm_cell/clip_by_value_1/y?
$LSTM_lay_0/lstm_cell/clip_by_value_1Maximum0LSTM_lay_0/lstm_cell/clip_by_value_1/Minimum:z:0/LSTM_lay_0/lstm_cell/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:??????????2&
$LSTM_lay_0/lstm_cell/clip_by_value_1?
LSTM_lay_0/lstm_cell/mul_2Mul(LSTM_lay_0/lstm_cell/clip_by_value_1:z:0LSTM_lay_0/zeros_1:output:0*
T0*(
_output_shapes
:??????????2
LSTM_lay_0/lstm_cell/mul_2?
%LSTM_lay_0/lstm_cell/ReadVariableOp_2ReadVariableOp,lstm_lay_0_lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02'
%LSTM_lay_0/lstm_cell/ReadVariableOp_2?
*LSTM_lay_0/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2,
*LSTM_lay_0/lstm_cell/strided_slice_2/stack?
,LSTM_lay_0/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2.
,LSTM_lay_0/lstm_cell/strided_slice_2/stack_1?
,LSTM_lay_0/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2.
,LSTM_lay_0/lstm_cell/strided_slice_2/stack_2?
$LSTM_lay_0/lstm_cell/strided_slice_2StridedSlice-LSTM_lay_0/lstm_cell/ReadVariableOp_2:value:03LSTM_lay_0/lstm_cell/strided_slice_2/stack:output:05LSTM_lay_0/lstm_cell/strided_slice_2/stack_1:output:05LSTM_lay_0/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2&
$LSTM_lay_0/lstm_cell/strided_slice_2?
LSTM_lay_0/lstm_cell/MatMul_6MatMulLSTM_lay_0/zeros:output:0-LSTM_lay_0/lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
LSTM_lay_0/lstm_cell/MatMul_6?
LSTM_lay_0/lstm_cell/add_4AddV2'LSTM_lay_0/lstm_cell/BiasAdd_2:output:0'LSTM_lay_0/lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
LSTM_lay_0/lstm_cell/add_4?
LSTM_lay_0/lstm_cell/SigmoidSigmoidLSTM_lay_0/lstm_cell/add_4:z:0*
T0*(
_output_shapes
:??????????2
LSTM_lay_0/lstm_cell/Sigmoid?
LSTM_lay_0/lstm_cell/mul_3Mul&LSTM_lay_0/lstm_cell/clip_by_value:z:0 LSTM_lay_0/lstm_cell/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
LSTM_lay_0/lstm_cell/mul_3?
LSTM_lay_0/lstm_cell/add_5AddV2LSTM_lay_0/lstm_cell/mul_2:z:0LSTM_lay_0/lstm_cell/mul_3:z:0*
T0*(
_output_shapes
:??????????2
LSTM_lay_0/lstm_cell/add_5?
%LSTM_lay_0/lstm_cell/ReadVariableOp_3ReadVariableOp,lstm_lay_0_lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02'
%LSTM_lay_0/lstm_cell/ReadVariableOp_3?
*LSTM_lay_0/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2,
*LSTM_lay_0/lstm_cell/strided_slice_3/stack?
,LSTM_lay_0/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2.
,LSTM_lay_0/lstm_cell/strided_slice_3/stack_1?
,LSTM_lay_0/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2.
,LSTM_lay_0/lstm_cell/strided_slice_3/stack_2?
$LSTM_lay_0/lstm_cell/strided_slice_3StridedSlice-LSTM_lay_0/lstm_cell/ReadVariableOp_3:value:03LSTM_lay_0/lstm_cell/strided_slice_3/stack:output:05LSTM_lay_0/lstm_cell/strided_slice_3/stack_1:output:05LSTM_lay_0/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2&
$LSTM_lay_0/lstm_cell/strided_slice_3?
LSTM_lay_0/lstm_cell/MatMul_7MatMulLSTM_lay_0/zeros:output:0-LSTM_lay_0/lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
LSTM_lay_0/lstm_cell/MatMul_7?
LSTM_lay_0/lstm_cell/add_6AddV2'LSTM_lay_0/lstm_cell/BiasAdd_3:output:0'LSTM_lay_0/lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
LSTM_lay_0/lstm_cell/add_6?
LSTM_lay_0/lstm_cell/Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
LSTM_lay_0/lstm_cell/Const_6?
LSTM_lay_0/lstm_cell/Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
LSTM_lay_0/lstm_cell/Const_7?
LSTM_lay_0/lstm_cell/Mul_4MulLSTM_lay_0/lstm_cell/add_6:z:0%LSTM_lay_0/lstm_cell/Const_6:output:0*
T0*(
_output_shapes
:??????????2
LSTM_lay_0/lstm_cell/Mul_4?
LSTM_lay_0/lstm_cell/Add_7AddLSTM_lay_0/lstm_cell/Mul_4:z:0%LSTM_lay_0/lstm_cell/Const_7:output:0*
T0*(
_output_shapes
:??????????2
LSTM_lay_0/lstm_cell/Add_7?
.LSTM_lay_0/lstm_cell/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??20
.LSTM_lay_0/lstm_cell/clip_by_value_2/Minimum/y?
,LSTM_lay_0/lstm_cell/clip_by_value_2/MinimumMinimumLSTM_lay_0/lstm_cell/Add_7:z:07LSTM_lay_0/lstm_cell/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:??????????2.
,LSTM_lay_0/lstm_cell/clip_by_value_2/Minimum?
&LSTM_lay_0/lstm_cell/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2(
&LSTM_lay_0/lstm_cell/clip_by_value_2/y?
$LSTM_lay_0/lstm_cell/clip_by_value_2Maximum0LSTM_lay_0/lstm_cell/clip_by_value_2/Minimum:z:0/LSTM_lay_0/lstm_cell/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:??????????2&
$LSTM_lay_0/lstm_cell/clip_by_value_2?
LSTM_lay_0/lstm_cell/Sigmoid_1SigmoidLSTM_lay_0/lstm_cell/add_5:z:0*
T0*(
_output_shapes
:??????????2 
LSTM_lay_0/lstm_cell/Sigmoid_1?
LSTM_lay_0/lstm_cell/mul_5Mul(LSTM_lay_0/lstm_cell/clip_by_value_2:z:0"LSTM_lay_0/lstm_cell/Sigmoid_1:y:0*
T0*(
_output_shapes
:??????????2
LSTM_lay_0/lstm_cell/mul_5?
(LSTM_lay_0/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2*
(LSTM_lay_0/TensorArrayV2_1/element_shape?
LSTM_lay_0/TensorArrayV2_1TensorListReserve1LSTM_lay_0/TensorArrayV2_1/element_shape:output:0#LSTM_lay_0/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
LSTM_lay_0/TensorArrayV2_1d
LSTM_lay_0/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
LSTM_lay_0/time?
#LSTM_lay_0/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#LSTM_lay_0/while/maximum_iterations?
LSTM_lay_0/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
LSTM_lay_0/while/loop_counter?
LSTM_lay_0/whileWhile&LSTM_lay_0/while/loop_counter:output:0,LSTM_lay_0/while/maximum_iterations:output:0LSTM_lay_0/time:output:0#LSTM_lay_0/TensorArrayV2_1:handle:0LSTM_lay_0/zeros:output:0LSTM_lay_0/zeros_1:output:0#LSTM_lay_0/strided_slice_1:output:0BLSTM_lay_0/TensorArrayUnstack/TensorListFromTensor:output_handle:02lstm_lay_0_lstm_cell_split_readvariableop_resource4lstm_lay_0_lstm_cell_split_1_readvariableop_resource,lstm_lay_0_lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*(
body R
LSTM_lay_0_while_body_120604*(
cond R
LSTM_lay_0_while_cond_120603*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
LSTM_lay_0/while?
;LSTM_lay_0/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2=
;LSTM_lay_0/TensorArrayV2Stack/TensorListStack/element_shape?
-LSTM_lay_0/TensorArrayV2Stack/TensorListStackTensorListStackLSTM_lay_0/while:output:3DLSTM_lay_0/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype02/
-LSTM_lay_0/TensorArrayV2Stack/TensorListStack?
 LSTM_lay_0/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2"
 LSTM_lay_0/strided_slice_3/stack?
"LSTM_lay_0/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"LSTM_lay_0/strided_slice_3/stack_1?
"LSTM_lay_0/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"LSTM_lay_0/strided_slice_3/stack_2?
LSTM_lay_0/strided_slice_3StridedSlice6LSTM_lay_0/TensorArrayV2Stack/TensorListStack:tensor:0)LSTM_lay_0/strided_slice_3/stack:output:0+LSTM_lay_0/strided_slice_3/stack_1:output:0+LSTM_lay_0/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
LSTM_lay_0/strided_slice_3?
LSTM_lay_0/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
LSTM_lay_0/transpose_1/perm?
LSTM_lay_0/transpose_1	Transpose6LSTM_lay_0/TensorArrayV2Stack/TensorListStack:tensor:0$LSTM_lay_0/transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2
LSTM_lay_0/transpose_1?
Dropout_lay_0/IdentityIdentity#LSTM_lay_0/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
Dropout_lay_0/Identity?
!Dense_lay_1/MatMul/ReadVariableOpReadVariableOp*dense_lay_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02#
!Dense_lay_1/MatMul/ReadVariableOp?
Dense_lay_1/MatMulMatMulDropout_lay_0/Identity:output:0)Dense_lay_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Dense_lay_1/MatMul?
"Dense_lay_1/BiasAdd/ReadVariableOpReadVariableOp+dense_lay_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02$
"Dense_lay_1/BiasAdd/ReadVariableOp?
Dense_lay_1/BiasAddBiasAddDense_lay_1/MatMul:product:0*Dense_lay_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Dense_lay_1/BiasAdd}
Dense_lay_1/ReluReluDense_lay_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Dense_lay_1/Relu?
 Output_lay/MatMul/ReadVariableOpReadVariableOp)output_lay_matmul_readvariableop_resource*
_output_shapes
:	?`*
dtype02"
 Output_lay/MatMul/ReadVariableOp?
Output_lay/MatMulMatMulDense_lay_1/Relu:activations:0(Output_lay/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????`2
Output_lay/MatMul?
!Output_lay/BiasAdd/ReadVariableOpReadVariableOp*output_lay_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype02#
!Output_lay/BiasAdd/ReadVariableOp?
Output_lay/BiasAddBiasAddOutput_lay/MatMul:product:0)Output_lay/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????`2
Output_lay/BiasAdd?
IdentityIdentityOutput_lay/BiasAdd:output:0#^Dense_lay_1/BiasAdd/ReadVariableOp"^Dense_lay_1/MatMul/ReadVariableOp$^LSTM_lay_0/lstm_cell/ReadVariableOp&^LSTM_lay_0/lstm_cell/ReadVariableOp_1&^LSTM_lay_0/lstm_cell/ReadVariableOp_2&^LSTM_lay_0/lstm_cell/ReadVariableOp_3*^LSTM_lay_0/lstm_cell/split/ReadVariableOp,^LSTM_lay_0/lstm_cell/split_1/ReadVariableOp^LSTM_lay_0/while"^Output_lay/BiasAdd/ReadVariableOp!^Output_lay/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????`2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:??????????:::::::2H
"Dense_lay_1/BiasAdd/ReadVariableOp"Dense_lay_1/BiasAdd/ReadVariableOp2F
!Dense_lay_1/MatMul/ReadVariableOp!Dense_lay_1/MatMul/ReadVariableOp2J
#LSTM_lay_0/lstm_cell/ReadVariableOp#LSTM_lay_0/lstm_cell/ReadVariableOp2N
%LSTM_lay_0/lstm_cell/ReadVariableOp_1%LSTM_lay_0/lstm_cell/ReadVariableOp_12N
%LSTM_lay_0/lstm_cell/ReadVariableOp_2%LSTM_lay_0/lstm_cell/ReadVariableOp_22N
%LSTM_lay_0/lstm_cell/ReadVariableOp_3%LSTM_lay_0/lstm_cell/ReadVariableOp_32V
)LSTM_lay_0/lstm_cell/split/ReadVariableOp)LSTM_lay_0/lstm_cell/split/ReadVariableOp2Z
+LSTM_lay_0/lstm_cell/split_1/ReadVariableOp+LSTM_lay_0/lstm_cell/split_1/ReadVariableOp2$
LSTM_lay_0/whileLSTM_lay_0/while2F
!Output_lay/BiasAdd/ReadVariableOp!Output_lay/BiasAdd/ReadVariableOp2D
 Output_lay/MatMul/ReadVariableOp Output_lay/MatMul/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
+__inference_LSTM_lay_0_layer_call_fn_121914
inputs_0
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_LSTM_lay_0_layer_call_and_return_conditional_losses_1193932
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:???????????????????:::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:???????????????????
"
_user_specified_name
inputs/0
?V
?
E__inference_lstm_cell_layer_call_and_return_conditional_losses_118811

inputs

states
states_1!
split_readvariableop_resource#
split_1_readvariableop_resource
readvariableop_resource
identity

identity_1

identity_2??ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?ReadVariableOp_3?split/ReadVariableOp?split_1/ReadVariableOpP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource* 
_output_shapes
:
??*
dtype02
split/ReadVariableOp?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
??:
??:
??:
??*
	num_split2
splite
MatMulMatMulinputssplit:output:0*
T0*(
_output_shapes
:??????????2
MatMuli
MatMul_1MatMulinputssplit:output:1*
T0*(
_output_shapes
:??????????2

MatMul_1i
MatMul_2MatMulinputssplit:output:2*
T0*(
_output_shapes
:??????????2

MatMul_2i
MatMul_3MatMulinputssplit:output:3*
T0*(
_output_shapes
:??????????2

MatMul_3T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_1/split_dim?
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:?*
dtype02
split_1/ReadVariableOp?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2	
split_1t
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*(
_output_shapes
:??????????2	
BiasAddz
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*(
_output_shapes
:??????????2
	BiasAdd_1z
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*(
_output_shapes
:??????????2
	BiasAdd_2z
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*(
_output_shapes
:??????????2
	BiasAdd_3z
ReadVariableOpReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype02
ReadVariableOp{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
strided_sliceq
MatMul_4MatMulstatesstrided_slice:output:0*
T0*(
_output_shapes
:??????????2

MatMul_4l
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
addW
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_2W
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_3_
MulMuladd:z:0Const_2:output:0*
T0*(
_output_shapes
:??????????2
Mulc
Add_1AddMul:z:0Const_3:output:0*
T0*(
_output_shapes
:??????????2
Add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:??????????2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*(
_output_shapes
:??????????2
clip_by_value~
ReadVariableOp_1ReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype02
ReadVariableOp_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
strided_slice_1s
MatMul_5MatMulstatesstrided_slice_1:output:0*
T0*(
_output_shapes
:??????????2

MatMul_5r
add_2AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
add_2W
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_4W
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_5e
Mul_1Mul	add_2:z:0Const_4:output:0*
T0*(
_output_shapes
:??????????2
Mul_1e
Add_3Add	Mul_1:z:0Const_5:output:0*
T0*(
_output_shapes
:??????????2
Add_3{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_1/Minimum/y?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:??????????2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*(
_output_shapes
:??????????2
clip_by_value_1g
mul_2Mulclip_by_value_1:z:0states_1*
T0*(
_output_shapes
:??????????2
mul_2~
ReadVariableOp_2ReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype02
ReadVariableOp_2
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2
strided_slice_2/stack_1?
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2?
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
strided_slice_2s
MatMul_6MatMulstatesstrided_slice_2:output:0*
T0*(
_output_shapes
:??????????2

MatMul_6r
add_4AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
add_4[
SigmoidSigmoid	add_4:z:0*
T0*(
_output_shapes
:??????????2	
Sigmoidh
mul_3Mulclip_by_value:z:0Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
mul_3`
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*(
_output_shapes
:??????????2
add_5~
ReadVariableOp_3ReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype02
ReadVariableOp_3
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2
strided_slice_3/stack?
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack_1?
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2?
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
strided_slice_3s
MatMul_7MatMulstatesstrided_slice_3:output:0*
T0*(
_output_shapes
:??????????2

MatMul_7r
add_6AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
add_6W
Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_6W
Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_7e
Mul_4Mul	add_6:z:0Const_6:output:0*
T0*(
_output_shapes
:??????????2
Mul_4e
Add_7Add	Mul_4:z:0Const_7:output:0*
T0*(
_output_shapes
:??????????2
Add_7{
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_2/Minimum/y?
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:??????????2
clip_by_value_2/Minimumk
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_2/y?
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*(
_output_shapes
:??????????2
clip_by_value_2_
	Sigmoid_1Sigmoid	add_5:z:0*
T0*(
_output_shapes
:??????????2
	Sigmoid_1l
mul_5Mulclip_by_value_2:z:0Sigmoid_1:y:0*
T0*(
_output_shapes
:??????????2
mul_5?
IdentityIdentity	mul_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity	mul_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity_1?

Identity_2Identity	add_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*[
_input_shapesJ
H:??????????:??????????:??????????:::2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_namestates:PL
(
_output_shapes
:??????????
 
_user_specified_namestates
?

?
LSTM_lay_0_while_cond_1206032
.lstm_lay_0_while_lstm_lay_0_while_loop_counter8
4lstm_lay_0_while_lstm_lay_0_while_maximum_iterations 
lstm_lay_0_while_placeholder"
lstm_lay_0_while_placeholder_1"
lstm_lay_0_while_placeholder_2"
lstm_lay_0_while_placeholder_34
0lstm_lay_0_while_less_lstm_lay_0_strided_slice_1J
Flstm_lay_0_while_lstm_lay_0_while_cond_120603___redundant_placeholder0J
Flstm_lay_0_while_lstm_lay_0_while_cond_120603___redundant_placeholder1J
Flstm_lay_0_while_lstm_lay_0_while_cond_120603___redundant_placeholder2J
Flstm_lay_0_while_lstm_lay_0_while_cond_120603___redundant_placeholder3
lstm_lay_0_while_identity
?
LSTM_lay_0/while/LessLesslstm_lay_0_while_placeholder0lstm_lay_0_while_less_lstm_lay_0_strided_slice_1*
T0*
_output_shapes
: 2
LSTM_lay_0/while/Less~
LSTM_lay_0/while/IdentityIdentityLSTM_lay_0/while/Less:z:0*
T0
*
_output_shapes
: 2
LSTM_lay_0/while/Identity"?
lstm_lay_0_while_identity"LSTM_lay_0/while/Identity:output:0*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?$
?
while_body_119194
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_lstm_cell_119218_0
while_lstm_cell_119220_0
while_lstm_cell_119222_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_lstm_cell_119218
while_lstm_cell_119220
while_lstm_cell_119222??'while/lstm_cell/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
'while/lstm_cell/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_119218_0while_lstm_cell_119220_0while_lstm_cell_119222_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_lstm_cell_layer_call_and_return_conditional_losses_1188112)
'while/lstm_cell/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder0while/lstm_cell/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:0(^while/lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations(^while/lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0(^while/lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity0while/lstm_cell/StatefulPartitionedCall:output:1(^while/lstm_cell/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identity0while/lstm_cell/StatefulPartitionedCall:output:2(^while/lstm_cell/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"2
while_lstm_cell_119218while_lstm_cell_119218_0"2
while_lstm_cell_119220while_lstm_cell_119220_0"2
while_lstm_cell_119222while_lstm_cell_119222_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :??????????:??????????: : :::2R
'while/lstm_cell/StatefulPartitionedCall'while/lstm_cell/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
+__inference_LSTM_lay_0_layer_call_fn_121356

inputs
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_LSTM_lay_0_layer_call_and_return_conditional_losses_1199412
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????:::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
+__inference_sequential_layer_call_fn_120119
lstm_lay_0_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalllstm_lay_0_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????`*)
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_1201022
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????`2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:??????????:::::::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
,
_output_shapes
:??????????
*
_user_specified_nameLSTM_lay_0_input
?
?
F__inference_sequential_layer_call_and_return_conditional_losses_120102

inputs
lstm_lay_0_120083
lstm_lay_0_120085
lstm_lay_0_120087
dense_lay_1_120091
dense_lay_1_120093
output_lay_120096
output_lay_120098
identity??#Dense_lay_1/StatefulPartitionedCall?%Dropout_lay_0/StatefulPartitionedCall?"LSTM_lay_0/StatefulPartitionedCall?"Output_lay/StatefulPartitionedCall?
"LSTM_lay_0/StatefulPartitionedCallStatefulPartitionedCallinputslstm_lay_0_120083lstm_lay_0_120085lstm_lay_0_120087*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_LSTM_lay_0_layer_call_and_return_conditional_losses_1196732$
"LSTM_lay_0/StatefulPartitionedCall?
%Dropout_lay_0/StatefulPartitionedCallStatefulPartitionedCall+LSTM_lay_0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_Dropout_lay_0_layer_call_and_return_conditional_losses_1199832'
%Dropout_lay_0/StatefulPartitionedCall?
#Dense_lay_1/StatefulPartitionedCallStatefulPartitionedCall.Dropout_lay_0/StatefulPartitionedCall:output:0dense_lay_1_120091dense_lay_1_120093*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_Dense_lay_1_layer_call_and_return_conditional_losses_1200122%
#Dense_lay_1/StatefulPartitionedCall?
"Output_lay/StatefulPartitionedCallStatefulPartitionedCall,Dense_lay_1/StatefulPartitionedCall:output:0output_lay_120096output_lay_120098*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????`*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_Output_lay_layer_call_and_return_conditional_losses_1200382$
"Output_lay/StatefulPartitionedCall?
IdentityIdentity+Output_lay/StatefulPartitionedCall:output:0$^Dense_lay_1/StatefulPartitionedCall&^Dropout_lay_0/StatefulPartitionedCall#^LSTM_lay_0/StatefulPartitionedCall#^Output_lay/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????`2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:??????????:::::::2J
#Dense_lay_1/StatefulPartitionedCall#Dense_lay_1/StatefulPartitionedCall2N
%Dropout_lay_0/StatefulPartitionedCall%Dropout_lay_0/StatefulPartitionedCall2H
"LSTM_lay_0/StatefulPartitionedCall"LSTM_lay_0/StatefulPartitionedCall2H
"Output_lay/StatefulPartitionedCall"Output_lay/StatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
LSTM_lay_0_while_cond_1203142
.lstm_lay_0_while_lstm_lay_0_while_loop_counter8
4lstm_lay_0_while_lstm_lay_0_while_maximum_iterations 
lstm_lay_0_while_placeholder"
lstm_lay_0_while_placeholder_1"
lstm_lay_0_while_placeholder_2"
lstm_lay_0_while_placeholder_34
0lstm_lay_0_while_less_lstm_lay_0_strided_slice_1J
Flstm_lay_0_while_lstm_lay_0_while_cond_120314___redundant_placeholder0J
Flstm_lay_0_while_lstm_lay_0_while_cond_120314___redundant_placeholder1J
Flstm_lay_0_while_lstm_lay_0_while_cond_120314___redundant_placeholder2J
Flstm_lay_0_while_lstm_lay_0_while_cond_120314___redundant_placeholder3
lstm_lay_0_while_identity
?
LSTM_lay_0/while/LessLesslstm_lay_0_while_placeholder0lstm_lay_0_while_less_lstm_lay_0_strided_slice_1*
T0*
_output_shapes
: 2
LSTM_lay_0/while/Less~
LSTM_lay_0/while/IdentityIdentityLSTM_lay_0/while/Less:z:0*
T0
*
_output_shapes
: 2
LSTM_lay_0/while/Identity"?
lstm_lay_0_while_identity"LSTM_lay_0/while/Identity:output:0*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
??
?
F__inference_sequential_layer_call_and_return_conditional_losses_120478

inputs6
2lstm_lay_0_lstm_cell_split_readvariableop_resource8
4lstm_lay_0_lstm_cell_split_1_readvariableop_resource0
,lstm_lay_0_lstm_cell_readvariableop_resource.
*dense_lay_1_matmul_readvariableop_resource/
+dense_lay_1_biasadd_readvariableop_resource-
)output_lay_matmul_readvariableop_resource.
*output_lay_biasadd_readvariableop_resource
identity??"Dense_lay_1/BiasAdd/ReadVariableOp?!Dense_lay_1/MatMul/ReadVariableOp?#LSTM_lay_0/lstm_cell/ReadVariableOp?%LSTM_lay_0/lstm_cell/ReadVariableOp_1?%LSTM_lay_0/lstm_cell/ReadVariableOp_2?%LSTM_lay_0/lstm_cell/ReadVariableOp_3?)LSTM_lay_0/lstm_cell/split/ReadVariableOp?+LSTM_lay_0/lstm_cell/split_1/ReadVariableOp?LSTM_lay_0/while?!Output_lay/BiasAdd/ReadVariableOp? Output_lay/MatMul/ReadVariableOpZ
LSTM_lay_0/ShapeShapeinputs*
T0*
_output_shapes
:2
LSTM_lay_0/Shape?
LSTM_lay_0/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
LSTM_lay_0/strided_slice/stack?
 LSTM_lay_0/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 LSTM_lay_0/strided_slice/stack_1?
 LSTM_lay_0/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 LSTM_lay_0/strided_slice/stack_2?
LSTM_lay_0/strided_sliceStridedSliceLSTM_lay_0/Shape:output:0'LSTM_lay_0/strided_slice/stack:output:0)LSTM_lay_0/strided_slice/stack_1:output:0)LSTM_lay_0/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
LSTM_lay_0/strided_slices
LSTM_lay_0/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
LSTM_lay_0/zeros/mul/y?
LSTM_lay_0/zeros/mulMul!LSTM_lay_0/strided_slice:output:0LSTM_lay_0/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
LSTM_lay_0/zeros/mulu
LSTM_lay_0/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
LSTM_lay_0/zeros/Less/y?
LSTM_lay_0/zeros/LessLessLSTM_lay_0/zeros/mul:z:0 LSTM_lay_0/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
LSTM_lay_0/zeros/Lessy
LSTM_lay_0/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
LSTM_lay_0/zeros/packed/1?
LSTM_lay_0/zeros/packedPack!LSTM_lay_0/strided_slice:output:0"LSTM_lay_0/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
LSTM_lay_0/zeros/packedu
LSTM_lay_0/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
LSTM_lay_0/zeros/Const?
LSTM_lay_0/zerosFill LSTM_lay_0/zeros/packed:output:0LSTM_lay_0/zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
LSTM_lay_0/zerosw
LSTM_lay_0/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
LSTM_lay_0/zeros_1/mul/y?
LSTM_lay_0/zeros_1/mulMul!LSTM_lay_0/strided_slice:output:0!LSTM_lay_0/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
LSTM_lay_0/zeros_1/muly
LSTM_lay_0/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
LSTM_lay_0/zeros_1/Less/y?
LSTM_lay_0/zeros_1/LessLessLSTM_lay_0/zeros_1/mul:z:0"LSTM_lay_0/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
LSTM_lay_0/zeros_1/Less}
LSTM_lay_0/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
LSTM_lay_0/zeros_1/packed/1?
LSTM_lay_0/zeros_1/packedPack!LSTM_lay_0/strided_slice:output:0$LSTM_lay_0/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
LSTM_lay_0/zeros_1/packedy
LSTM_lay_0/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
LSTM_lay_0/zeros_1/Const?
LSTM_lay_0/zeros_1Fill"LSTM_lay_0/zeros_1/packed:output:0!LSTM_lay_0/zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2
LSTM_lay_0/zeros_1?
LSTM_lay_0/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
LSTM_lay_0/transpose/perm?
LSTM_lay_0/transpose	Transposeinputs"LSTM_lay_0/transpose/perm:output:0*
T0*,
_output_shapes
:??????????2
LSTM_lay_0/transposep
LSTM_lay_0/Shape_1ShapeLSTM_lay_0/transpose:y:0*
T0*
_output_shapes
:2
LSTM_lay_0/Shape_1?
 LSTM_lay_0/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 LSTM_lay_0/strided_slice_1/stack?
"LSTM_lay_0/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"LSTM_lay_0/strided_slice_1/stack_1?
"LSTM_lay_0/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"LSTM_lay_0/strided_slice_1/stack_2?
LSTM_lay_0/strided_slice_1StridedSliceLSTM_lay_0/Shape_1:output:0)LSTM_lay_0/strided_slice_1/stack:output:0+LSTM_lay_0/strided_slice_1/stack_1:output:0+LSTM_lay_0/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
LSTM_lay_0/strided_slice_1?
&LSTM_lay_0/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2(
&LSTM_lay_0/TensorArrayV2/element_shape?
LSTM_lay_0/TensorArrayV2TensorListReserve/LSTM_lay_0/TensorArrayV2/element_shape:output:0#LSTM_lay_0/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
LSTM_lay_0/TensorArrayV2?
@LSTM_lay_0/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2B
@LSTM_lay_0/TensorArrayUnstack/TensorListFromTensor/element_shape?
2LSTM_lay_0/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorLSTM_lay_0/transpose:y:0ILSTM_lay_0/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type024
2LSTM_lay_0/TensorArrayUnstack/TensorListFromTensor?
 LSTM_lay_0/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 LSTM_lay_0/strided_slice_2/stack?
"LSTM_lay_0/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"LSTM_lay_0/strided_slice_2/stack_1?
"LSTM_lay_0/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"LSTM_lay_0/strided_slice_2/stack_2?
LSTM_lay_0/strided_slice_2StridedSliceLSTM_lay_0/transpose:y:0)LSTM_lay_0/strided_slice_2/stack:output:0+LSTM_lay_0/strided_slice_2/stack_1:output:0+LSTM_lay_0/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
LSTM_lay_0/strided_slice_2z
LSTM_lay_0/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
LSTM_lay_0/lstm_cell/Const?
$LSTM_lay_0/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$LSTM_lay_0/lstm_cell/split/split_dim?
)LSTM_lay_0/lstm_cell/split/ReadVariableOpReadVariableOp2lstm_lay_0_lstm_cell_split_readvariableop_resource* 
_output_shapes
:
??*
dtype02+
)LSTM_lay_0/lstm_cell/split/ReadVariableOp?
LSTM_lay_0/lstm_cell/splitSplit-LSTM_lay_0/lstm_cell/split/split_dim:output:01LSTM_lay_0/lstm_cell/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
??:
??:
??:
??*
	num_split2
LSTM_lay_0/lstm_cell/split?
LSTM_lay_0/lstm_cell/MatMulMatMul#LSTM_lay_0/strided_slice_2:output:0#LSTM_lay_0/lstm_cell/split:output:0*
T0*(
_output_shapes
:??????????2
LSTM_lay_0/lstm_cell/MatMul?
LSTM_lay_0/lstm_cell/MatMul_1MatMul#LSTM_lay_0/strided_slice_2:output:0#LSTM_lay_0/lstm_cell/split:output:1*
T0*(
_output_shapes
:??????????2
LSTM_lay_0/lstm_cell/MatMul_1?
LSTM_lay_0/lstm_cell/MatMul_2MatMul#LSTM_lay_0/strided_slice_2:output:0#LSTM_lay_0/lstm_cell/split:output:2*
T0*(
_output_shapes
:??????????2
LSTM_lay_0/lstm_cell/MatMul_2?
LSTM_lay_0/lstm_cell/MatMul_3MatMul#LSTM_lay_0/strided_slice_2:output:0#LSTM_lay_0/lstm_cell/split:output:3*
T0*(
_output_shapes
:??????????2
LSTM_lay_0/lstm_cell/MatMul_3~
LSTM_lay_0/lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
LSTM_lay_0/lstm_cell/Const_1?
&LSTM_lay_0/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2(
&LSTM_lay_0/lstm_cell/split_1/split_dim?
+LSTM_lay_0/lstm_cell/split_1/ReadVariableOpReadVariableOp4lstm_lay_0_lstm_cell_split_1_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+LSTM_lay_0/lstm_cell/split_1/ReadVariableOp?
LSTM_lay_0/lstm_cell/split_1Split/LSTM_lay_0/lstm_cell/split_1/split_dim:output:03LSTM_lay_0/lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2
LSTM_lay_0/lstm_cell/split_1?
LSTM_lay_0/lstm_cell/BiasAddBiasAdd%LSTM_lay_0/lstm_cell/MatMul:product:0%LSTM_lay_0/lstm_cell/split_1:output:0*
T0*(
_output_shapes
:??????????2
LSTM_lay_0/lstm_cell/BiasAdd?
LSTM_lay_0/lstm_cell/BiasAdd_1BiasAdd'LSTM_lay_0/lstm_cell/MatMul_1:product:0%LSTM_lay_0/lstm_cell/split_1:output:1*
T0*(
_output_shapes
:??????????2 
LSTM_lay_0/lstm_cell/BiasAdd_1?
LSTM_lay_0/lstm_cell/BiasAdd_2BiasAdd'LSTM_lay_0/lstm_cell/MatMul_2:product:0%LSTM_lay_0/lstm_cell/split_1:output:2*
T0*(
_output_shapes
:??????????2 
LSTM_lay_0/lstm_cell/BiasAdd_2?
LSTM_lay_0/lstm_cell/BiasAdd_3BiasAdd'LSTM_lay_0/lstm_cell/MatMul_3:product:0%LSTM_lay_0/lstm_cell/split_1:output:3*
T0*(
_output_shapes
:??????????2 
LSTM_lay_0/lstm_cell/BiasAdd_3?
#LSTM_lay_0/lstm_cell/ReadVariableOpReadVariableOp,lstm_lay_0_lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02%
#LSTM_lay_0/lstm_cell/ReadVariableOp?
(LSTM_lay_0/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2*
(LSTM_lay_0/lstm_cell/strided_slice/stack?
*LSTM_lay_0/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2,
*LSTM_lay_0/lstm_cell/strided_slice/stack_1?
*LSTM_lay_0/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*LSTM_lay_0/lstm_cell/strided_slice/stack_2?
"LSTM_lay_0/lstm_cell/strided_sliceStridedSlice+LSTM_lay_0/lstm_cell/ReadVariableOp:value:01LSTM_lay_0/lstm_cell/strided_slice/stack:output:03LSTM_lay_0/lstm_cell/strided_slice/stack_1:output:03LSTM_lay_0/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2$
"LSTM_lay_0/lstm_cell/strided_slice?
LSTM_lay_0/lstm_cell/MatMul_4MatMulLSTM_lay_0/zeros:output:0+LSTM_lay_0/lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
LSTM_lay_0/lstm_cell/MatMul_4?
LSTM_lay_0/lstm_cell/addAddV2%LSTM_lay_0/lstm_cell/BiasAdd:output:0'LSTM_lay_0/lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
LSTM_lay_0/lstm_cell/add?
LSTM_lay_0/lstm_cell/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
LSTM_lay_0/lstm_cell/Const_2?
LSTM_lay_0/lstm_cell/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
LSTM_lay_0/lstm_cell/Const_3?
LSTM_lay_0/lstm_cell/MulMulLSTM_lay_0/lstm_cell/add:z:0%LSTM_lay_0/lstm_cell/Const_2:output:0*
T0*(
_output_shapes
:??????????2
LSTM_lay_0/lstm_cell/Mul?
LSTM_lay_0/lstm_cell/Add_1AddLSTM_lay_0/lstm_cell/Mul:z:0%LSTM_lay_0/lstm_cell/Const_3:output:0*
T0*(
_output_shapes
:??????????2
LSTM_lay_0/lstm_cell/Add_1?
,LSTM_lay_0/lstm_cell/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2.
,LSTM_lay_0/lstm_cell/clip_by_value/Minimum/y?
*LSTM_lay_0/lstm_cell/clip_by_value/MinimumMinimumLSTM_lay_0/lstm_cell/Add_1:z:05LSTM_lay_0/lstm_cell/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:??????????2,
*LSTM_lay_0/lstm_cell/clip_by_value/Minimum?
$LSTM_lay_0/lstm_cell/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2&
$LSTM_lay_0/lstm_cell/clip_by_value/y?
"LSTM_lay_0/lstm_cell/clip_by_valueMaximum.LSTM_lay_0/lstm_cell/clip_by_value/Minimum:z:0-LSTM_lay_0/lstm_cell/clip_by_value/y:output:0*
T0*(
_output_shapes
:??????????2$
"LSTM_lay_0/lstm_cell/clip_by_value?
%LSTM_lay_0/lstm_cell/ReadVariableOp_1ReadVariableOp,lstm_lay_0_lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02'
%LSTM_lay_0/lstm_cell/ReadVariableOp_1?
*LSTM_lay_0/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2,
*LSTM_lay_0/lstm_cell/strided_slice_1/stack?
,LSTM_lay_0/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2.
,LSTM_lay_0/lstm_cell/strided_slice_1/stack_1?
,LSTM_lay_0/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2.
,LSTM_lay_0/lstm_cell/strided_slice_1/stack_2?
$LSTM_lay_0/lstm_cell/strided_slice_1StridedSlice-LSTM_lay_0/lstm_cell/ReadVariableOp_1:value:03LSTM_lay_0/lstm_cell/strided_slice_1/stack:output:05LSTM_lay_0/lstm_cell/strided_slice_1/stack_1:output:05LSTM_lay_0/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2&
$LSTM_lay_0/lstm_cell/strided_slice_1?
LSTM_lay_0/lstm_cell/MatMul_5MatMulLSTM_lay_0/zeros:output:0-LSTM_lay_0/lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
LSTM_lay_0/lstm_cell/MatMul_5?
LSTM_lay_0/lstm_cell/add_2AddV2'LSTM_lay_0/lstm_cell/BiasAdd_1:output:0'LSTM_lay_0/lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
LSTM_lay_0/lstm_cell/add_2?
LSTM_lay_0/lstm_cell/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
LSTM_lay_0/lstm_cell/Const_4?
LSTM_lay_0/lstm_cell/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
LSTM_lay_0/lstm_cell/Const_5?
LSTM_lay_0/lstm_cell/Mul_1MulLSTM_lay_0/lstm_cell/add_2:z:0%LSTM_lay_0/lstm_cell/Const_4:output:0*
T0*(
_output_shapes
:??????????2
LSTM_lay_0/lstm_cell/Mul_1?
LSTM_lay_0/lstm_cell/Add_3AddLSTM_lay_0/lstm_cell/Mul_1:z:0%LSTM_lay_0/lstm_cell/Const_5:output:0*
T0*(
_output_shapes
:??????????2
LSTM_lay_0/lstm_cell/Add_3?
.LSTM_lay_0/lstm_cell/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??20
.LSTM_lay_0/lstm_cell/clip_by_value_1/Minimum/y?
,LSTM_lay_0/lstm_cell/clip_by_value_1/MinimumMinimumLSTM_lay_0/lstm_cell/Add_3:z:07LSTM_lay_0/lstm_cell/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:??????????2.
,LSTM_lay_0/lstm_cell/clip_by_value_1/Minimum?
&LSTM_lay_0/lstm_cell/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2(
&LSTM_lay_0/lstm_cell/clip_by_value_1/y?
$LSTM_lay_0/lstm_cell/clip_by_value_1Maximum0LSTM_lay_0/lstm_cell/clip_by_value_1/Minimum:z:0/LSTM_lay_0/lstm_cell/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:??????????2&
$LSTM_lay_0/lstm_cell/clip_by_value_1?
LSTM_lay_0/lstm_cell/mul_2Mul(LSTM_lay_0/lstm_cell/clip_by_value_1:z:0LSTM_lay_0/zeros_1:output:0*
T0*(
_output_shapes
:??????????2
LSTM_lay_0/lstm_cell/mul_2?
%LSTM_lay_0/lstm_cell/ReadVariableOp_2ReadVariableOp,lstm_lay_0_lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02'
%LSTM_lay_0/lstm_cell/ReadVariableOp_2?
*LSTM_lay_0/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2,
*LSTM_lay_0/lstm_cell/strided_slice_2/stack?
,LSTM_lay_0/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2.
,LSTM_lay_0/lstm_cell/strided_slice_2/stack_1?
,LSTM_lay_0/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2.
,LSTM_lay_0/lstm_cell/strided_slice_2/stack_2?
$LSTM_lay_0/lstm_cell/strided_slice_2StridedSlice-LSTM_lay_0/lstm_cell/ReadVariableOp_2:value:03LSTM_lay_0/lstm_cell/strided_slice_2/stack:output:05LSTM_lay_0/lstm_cell/strided_slice_2/stack_1:output:05LSTM_lay_0/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2&
$LSTM_lay_0/lstm_cell/strided_slice_2?
LSTM_lay_0/lstm_cell/MatMul_6MatMulLSTM_lay_0/zeros:output:0-LSTM_lay_0/lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
LSTM_lay_0/lstm_cell/MatMul_6?
LSTM_lay_0/lstm_cell/add_4AddV2'LSTM_lay_0/lstm_cell/BiasAdd_2:output:0'LSTM_lay_0/lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
LSTM_lay_0/lstm_cell/add_4?
LSTM_lay_0/lstm_cell/SigmoidSigmoidLSTM_lay_0/lstm_cell/add_4:z:0*
T0*(
_output_shapes
:??????????2
LSTM_lay_0/lstm_cell/Sigmoid?
LSTM_lay_0/lstm_cell/mul_3Mul&LSTM_lay_0/lstm_cell/clip_by_value:z:0 LSTM_lay_0/lstm_cell/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
LSTM_lay_0/lstm_cell/mul_3?
LSTM_lay_0/lstm_cell/add_5AddV2LSTM_lay_0/lstm_cell/mul_2:z:0LSTM_lay_0/lstm_cell/mul_3:z:0*
T0*(
_output_shapes
:??????????2
LSTM_lay_0/lstm_cell/add_5?
%LSTM_lay_0/lstm_cell/ReadVariableOp_3ReadVariableOp,lstm_lay_0_lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02'
%LSTM_lay_0/lstm_cell/ReadVariableOp_3?
*LSTM_lay_0/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2,
*LSTM_lay_0/lstm_cell/strided_slice_3/stack?
,LSTM_lay_0/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2.
,LSTM_lay_0/lstm_cell/strided_slice_3/stack_1?
,LSTM_lay_0/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2.
,LSTM_lay_0/lstm_cell/strided_slice_3/stack_2?
$LSTM_lay_0/lstm_cell/strided_slice_3StridedSlice-LSTM_lay_0/lstm_cell/ReadVariableOp_3:value:03LSTM_lay_0/lstm_cell/strided_slice_3/stack:output:05LSTM_lay_0/lstm_cell/strided_slice_3/stack_1:output:05LSTM_lay_0/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2&
$LSTM_lay_0/lstm_cell/strided_slice_3?
LSTM_lay_0/lstm_cell/MatMul_7MatMulLSTM_lay_0/zeros:output:0-LSTM_lay_0/lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
LSTM_lay_0/lstm_cell/MatMul_7?
LSTM_lay_0/lstm_cell/add_6AddV2'LSTM_lay_0/lstm_cell/BiasAdd_3:output:0'LSTM_lay_0/lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
LSTM_lay_0/lstm_cell/add_6?
LSTM_lay_0/lstm_cell/Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
LSTM_lay_0/lstm_cell/Const_6?
LSTM_lay_0/lstm_cell/Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
LSTM_lay_0/lstm_cell/Const_7?
LSTM_lay_0/lstm_cell/Mul_4MulLSTM_lay_0/lstm_cell/add_6:z:0%LSTM_lay_0/lstm_cell/Const_6:output:0*
T0*(
_output_shapes
:??????????2
LSTM_lay_0/lstm_cell/Mul_4?
LSTM_lay_0/lstm_cell/Add_7AddLSTM_lay_0/lstm_cell/Mul_4:z:0%LSTM_lay_0/lstm_cell/Const_7:output:0*
T0*(
_output_shapes
:??????????2
LSTM_lay_0/lstm_cell/Add_7?
.LSTM_lay_0/lstm_cell/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??20
.LSTM_lay_0/lstm_cell/clip_by_value_2/Minimum/y?
,LSTM_lay_0/lstm_cell/clip_by_value_2/MinimumMinimumLSTM_lay_0/lstm_cell/Add_7:z:07LSTM_lay_0/lstm_cell/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:??????????2.
,LSTM_lay_0/lstm_cell/clip_by_value_2/Minimum?
&LSTM_lay_0/lstm_cell/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2(
&LSTM_lay_0/lstm_cell/clip_by_value_2/y?
$LSTM_lay_0/lstm_cell/clip_by_value_2Maximum0LSTM_lay_0/lstm_cell/clip_by_value_2/Minimum:z:0/LSTM_lay_0/lstm_cell/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:??????????2&
$LSTM_lay_0/lstm_cell/clip_by_value_2?
LSTM_lay_0/lstm_cell/Sigmoid_1SigmoidLSTM_lay_0/lstm_cell/add_5:z:0*
T0*(
_output_shapes
:??????????2 
LSTM_lay_0/lstm_cell/Sigmoid_1?
LSTM_lay_0/lstm_cell/mul_5Mul(LSTM_lay_0/lstm_cell/clip_by_value_2:z:0"LSTM_lay_0/lstm_cell/Sigmoid_1:y:0*
T0*(
_output_shapes
:??????????2
LSTM_lay_0/lstm_cell/mul_5?
(LSTM_lay_0/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2*
(LSTM_lay_0/TensorArrayV2_1/element_shape?
LSTM_lay_0/TensorArrayV2_1TensorListReserve1LSTM_lay_0/TensorArrayV2_1/element_shape:output:0#LSTM_lay_0/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
LSTM_lay_0/TensorArrayV2_1d
LSTM_lay_0/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
LSTM_lay_0/time?
#LSTM_lay_0/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#LSTM_lay_0/while/maximum_iterations?
LSTM_lay_0/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
LSTM_lay_0/while/loop_counter?
LSTM_lay_0/whileWhile&LSTM_lay_0/while/loop_counter:output:0,LSTM_lay_0/while/maximum_iterations:output:0LSTM_lay_0/time:output:0#LSTM_lay_0/TensorArrayV2_1:handle:0LSTM_lay_0/zeros:output:0LSTM_lay_0/zeros_1:output:0#LSTM_lay_0/strided_slice_1:output:0BLSTM_lay_0/TensorArrayUnstack/TensorListFromTensor:output_handle:02lstm_lay_0_lstm_cell_split_readvariableop_resource4lstm_lay_0_lstm_cell_split_1_readvariableop_resource,lstm_lay_0_lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*(
body R
LSTM_lay_0_while_body_120315*(
cond R
LSTM_lay_0_while_cond_120314*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
LSTM_lay_0/while?
;LSTM_lay_0/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2=
;LSTM_lay_0/TensorArrayV2Stack/TensorListStack/element_shape?
-LSTM_lay_0/TensorArrayV2Stack/TensorListStackTensorListStackLSTM_lay_0/while:output:3DLSTM_lay_0/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype02/
-LSTM_lay_0/TensorArrayV2Stack/TensorListStack?
 LSTM_lay_0/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2"
 LSTM_lay_0/strided_slice_3/stack?
"LSTM_lay_0/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"LSTM_lay_0/strided_slice_3/stack_1?
"LSTM_lay_0/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"LSTM_lay_0/strided_slice_3/stack_2?
LSTM_lay_0/strided_slice_3StridedSlice6LSTM_lay_0/TensorArrayV2Stack/TensorListStack:tensor:0)LSTM_lay_0/strided_slice_3/stack:output:0+LSTM_lay_0/strided_slice_3/stack_1:output:0+LSTM_lay_0/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
LSTM_lay_0/strided_slice_3?
LSTM_lay_0/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
LSTM_lay_0/transpose_1/perm?
LSTM_lay_0/transpose_1	Transpose6LSTM_lay_0/TensorArrayV2Stack/TensorListStack:tensor:0$LSTM_lay_0/transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2
LSTM_lay_0/transpose_1
Dropout_lay_0/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
Dropout_lay_0/dropout/Const?
Dropout_lay_0/dropout/MulMul#LSTM_lay_0/strided_slice_3:output:0$Dropout_lay_0/dropout/Const:output:0*
T0*(
_output_shapes
:??????????2
Dropout_lay_0/dropout/Mul?
Dropout_lay_0/dropout/ShapeShape#LSTM_lay_0/strided_slice_3:output:0*
T0*
_output_shapes
:2
Dropout_lay_0/dropout/Shape?
2Dropout_lay_0/dropout/random_uniform/RandomUniformRandomUniform$Dropout_lay_0/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype024
2Dropout_lay_0/dropout/random_uniform/RandomUniform?
$Dropout_lay_0/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2&
$Dropout_lay_0/dropout/GreaterEqual/y?
"Dropout_lay_0/dropout/GreaterEqualGreaterEqual;Dropout_lay_0/dropout/random_uniform/RandomUniform:output:0-Dropout_lay_0/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2$
"Dropout_lay_0/dropout/GreaterEqual?
Dropout_lay_0/dropout/CastCast&Dropout_lay_0/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
Dropout_lay_0/dropout/Cast?
Dropout_lay_0/dropout/Mul_1MulDropout_lay_0/dropout/Mul:z:0Dropout_lay_0/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
Dropout_lay_0/dropout/Mul_1?
!Dense_lay_1/MatMul/ReadVariableOpReadVariableOp*dense_lay_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02#
!Dense_lay_1/MatMul/ReadVariableOp?
Dense_lay_1/MatMulMatMulDropout_lay_0/dropout/Mul_1:z:0)Dense_lay_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Dense_lay_1/MatMul?
"Dense_lay_1/BiasAdd/ReadVariableOpReadVariableOp+dense_lay_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02$
"Dense_lay_1/BiasAdd/ReadVariableOp?
Dense_lay_1/BiasAddBiasAddDense_lay_1/MatMul:product:0*Dense_lay_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Dense_lay_1/BiasAdd}
Dense_lay_1/ReluReluDense_lay_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Dense_lay_1/Relu?
 Output_lay/MatMul/ReadVariableOpReadVariableOp)output_lay_matmul_readvariableop_resource*
_output_shapes
:	?`*
dtype02"
 Output_lay/MatMul/ReadVariableOp?
Output_lay/MatMulMatMulDense_lay_1/Relu:activations:0(Output_lay/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????`2
Output_lay/MatMul?
!Output_lay/BiasAdd/ReadVariableOpReadVariableOp*output_lay_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype02#
!Output_lay/BiasAdd/ReadVariableOp?
Output_lay/BiasAddBiasAddOutput_lay/MatMul:product:0)Output_lay/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????`2
Output_lay/BiasAdd?
IdentityIdentityOutput_lay/BiasAdd:output:0#^Dense_lay_1/BiasAdd/ReadVariableOp"^Dense_lay_1/MatMul/ReadVariableOp$^LSTM_lay_0/lstm_cell/ReadVariableOp&^LSTM_lay_0/lstm_cell/ReadVariableOp_1&^LSTM_lay_0/lstm_cell/ReadVariableOp_2&^LSTM_lay_0/lstm_cell/ReadVariableOp_3*^LSTM_lay_0/lstm_cell/split/ReadVariableOp,^LSTM_lay_0/lstm_cell/split_1/ReadVariableOp^LSTM_lay_0/while"^Output_lay/BiasAdd/ReadVariableOp!^Output_lay/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????`2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:??????????:::::::2H
"Dense_lay_1/BiasAdd/ReadVariableOp"Dense_lay_1/BiasAdd/ReadVariableOp2F
!Dense_lay_1/MatMul/ReadVariableOp!Dense_lay_1/MatMul/ReadVariableOp2J
#LSTM_lay_0/lstm_cell/ReadVariableOp#LSTM_lay_0/lstm_cell/ReadVariableOp2N
%LSTM_lay_0/lstm_cell/ReadVariableOp_1%LSTM_lay_0/lstm_cell/ReadVariableOp_12N
%LSTM_lay_0/lstm_cell/ReadVariableOp_2%LSTM_lay_0/lstm_cell/ReadVariableOp_22N
%LSTM_lay_0/lstm_cell/ReadVariableOp_3%LSTM_lay_0/lstm_cell/ReadVariableOp_32V
)LSTM_lay_0/lstm_cell/split/ReadVariableOp)LSTM_lay_0/lstm_cell/split/ReadVariableOp2Z
+LSTM_lay_0/lstm_cell/split_1/ReadVariableOp+LSTM_lay_0/lstm_cell/split_1/ReadVariableOp2$
LSTM_lay_0/whileLSTM_lay_0/while2F
!Output_lay/BiasAdd/ReadVariableOp!Output_lay/BiasAdd/ReadVariableOp2D
 Output_lay/MatMul/ReadVariableOp Output_lay/MatMul/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
while_cond_121749
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_121749___redundant_placeholder04
0while_while_cond_121749___redundant_placeholder14
0while_while_cond_121749___redundant_placeholder24
0while_while_cond_121749___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?$
?
while_body_119325
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_lstm_cell_119349_0
while_lstm_cell_119351_0
while_lstm_cell_119353_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_lstm_cell_119349
while_lstm_cell_119351
while_lstm_cell_119353??'while/lstm_cell/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
'while/lstm_cell/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_119349_0while_lstm_cell_119351_0while_lstm_cell_119353_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_lstm_cell_layer_call_and_return_conditional_losses_1189022)
'while/lstm_cell/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder0while/lstm_cell/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:0(^while/lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations(^while/lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0(^while/lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity0while/lstm_cell/StatefulPartitionedCall:output:1(^while/lstm_cell/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identity0while/lstm_cell/StatefulPartitionedCall:output:2(^while/lstm_cell/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"2
while_lstm_cell_119349while_lstm_cell_119349_0"2
while_lstm_cell_119351while_lstm_cell_119351_0"2
while_lstm_cell_119353while_lstm_cell_119353_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :??????????:??????????: : :::2R
'while/lstm_cell/StatefulPartitionedCall'while/lstm_cell/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
g
.__inference_Dropout_lay_0_layer_call_fn_121936

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_Dropout_lay_0_layer_call_and_return_conditional_losses_1199832
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
ۡ
?
F__inference_LSTM_lay_0_layer_call_and_return_conditional_losses_121624
inputs_0+
'lstm_cell_split_readvariableop_resource-
)lstm_cell_split_1_readvariableop_resource%
!lstm_cell_readvariableop_resource
identity??lstm_cell/ReadVariableOp?lstm_cell/ReadVariableOp_1?lstm_cell/ReadVariableOp_2?lstm_cell/ReadVariableOp_3?lstm_cell/split/ReadVariableOp? lstm_cell/split_1/ReadVariableOp?whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_2d
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dim?
lstm_cell/split/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
lstm_cell/split/ReadVariableOp?
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
??:
??:
??:
??*
	num_split2
lstm_cell/split?
lstm_cell/MatMulMatMulstrided_slice_2:output:0lstm_cell/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul?
lstm_cell/MatMul_1MatMulstrided_slice_2:output:0lstm_cell/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_1?
lstm_cell/MatMul_2MatMulstrided_slice_2:output:0lstm_cell/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_2?
lstm_cell/MatMul_3MatMulstrided_slice_2:output:0lstm_cell/split:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_3h
lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Const_1|
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell/split_1/split_dim?
 lstm_cell/split_1/ReadVariableOpReadVariableOp)lstm_cell_split_1_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 lstm_cell/split_1/ReadVariableOp?
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2
lstm_cell/split_1?
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd?
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_1?
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_2?
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_3?
lstm_cell/ReadVariableOpReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp?
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
lstm_cell/strided_slice/stack?
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2!
lstm_cell/strided_slice/stack_1?
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
lstm_cell/strided_slice/stack_2?
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice?
lstm_cell/MatMul_4MatMulzeros:output:0 lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_4?
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/addk
lstm_cell/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
lstm_cell/Const_2k
lstm_cell/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm_cell/Const_3?
lstm_cell/MulMullstm_cell/add:z:0lstm_cell/Const_2:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Mul?
lstm_cell/Add_1Addlstm_cell/Mul:z:0lstm_cell/Const_3:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Add_1?
!lstm_cell/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2#
!lstm_cell/clip_by_value/Minimum/y?
lstm_cell/clip_by_value/MinimumMinimumlstm_cell/Add_1:z:0*lstm_cell/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:??????????2!
lstm_cell/clip_by_value/Minimum{
lstm_cell/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_cell/clip_by_value/y?
lstm_cell/clip_by_valueMaximum#lstm_cell/clip_by_value/Minimum:z:0"lstm_cell/clip_by_value/y:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/clip_by_value?
lstm_cell/ReadVariableOp_1ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_1?
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2!
lstm_cell/strided_slice_1/stack?
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_1/stack_1?
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_1/stack_2?
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_1?
lstm_cell/MatMul_5MatMulzeros:output:0"lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_5?
lstm_cell/add_2AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_2k
lstm_cell/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
lstm_cell/Const_4k
lstm_cell/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm_cell/Const_5?
lstm_cell/Mul_1Mullstm_cell/add_2:z:0lstm_cell/Const_4:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Mul_1?
lstm_cell/Add_3Addlstm_cell/Mul_1:z:0lstm_cell/Const_5:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Add_3?
#lstm_cell/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2%
#lstm_cell/clip_by_value_1/Minimum/y?
!lstm_cell/clip_by_value_1/MinimumMinimumlstm_cell/Add_3:z:0,lstm_cell/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:??????????2#
!lstm_cell/clip_by_value_1/Minimum
lstm_cell/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_cell/clip_by_value_1/y?
lstm_cell/clip_by_value_1Maximum%lstm_cell/clip_by_value_1/Minimum:z:0$lstm_cell/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/clip_by_value_1?
lstm_cell/mul_2Mullstm_cell/clip_by_value_1:z:0zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_2?
lstm_cell/ReadVariableOp_2ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_2?
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_2/stack?
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2#
!lstm_cell/strided_slice_2/stack_1?
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_2/stack_2?
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_2?
lstm_cell/MatMul_6MatMulzeros:output:0"lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_6?
lstm_cell/add_4AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_4y
lstm_cell/SigmoidSigmoidlstm_cell/add_4:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid?
lstm_cell/mul_3Mullstm_cell/clip_by_value:z:0lstm_cell/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_3?
lstm_cell/add_5AddV2lstm_cell/mul_2:z:0lstm_cell/mul_3:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_5?
lstm_cell/ReadVariableOp_3ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_3?
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2!
lstm_cell/strided_slice_3/stack?
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_3/stack_1?
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_3/stack_2?
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_3?
lstm_cell/MatMul_7MatMulzeros:output:0"lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_7?
lstm_cell/add_6AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_6k
lstm_cell/Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
lstm_cell/Const_6k
lstm_cell/Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm_cell/Const_7?
lstm_cell/Mul_4Mullstm_cell/add_6:z:0lstm_cell/Const_6:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Mul_4?
lstm_cell/Add_7Addlstm_cell/Mul_4:z:0lstm_cell/Const_7:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Add_7?
#lstm_cell/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2%
#lstm_cell/clip_by_value_2/Minimum/y?
!lstm_cell/clip_by_value_2/MinimumMinimumlstm_cell/Add_7:z:0,lstm_cell/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:??????????2#
!lstm_cell/clip_by_value_2/Minimum
lstm_cell/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_cell/clip_by_value_2/y?
lstm_cell/clip_by_value_2Maximum%lstm_cell/clip_by_value_2/Minimum:z:0$lstm_cell/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/clip_by_value_2}
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_5:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid_1?
lstm_cell/mul_5Mullstm_cell/clip_by_value_2:z:0lstm_cell/Sigmoid_1:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_5?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0'lstm_cell_split_readvariableop_resource)lstm_cell_split_1_readvariableop_resource!lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_121482*
condR
while_cond_121481*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
transpose_1?
IdentityIdentitystrided_slice_3:output:0^lstm_cell/ReadVariableOp^lstm_cell/ReadVariableOp_1^lstm_cell/ReadVariableOp_2^lstm_cell/ReadVariableOp_3^lstm_cell/split/ReadVariableOp!^lstm_cell/split_1/ReadVariableOp^while*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:???????????????????:::24
lstm_cell/ReadVariableOplstm_cell/ReadVariableOp28
lstm_cell/ReadVariableOp_1lstm_cell/ReadVariableOp_128
lstm_cell/ReadVariableOp_2lstm_cell/ReadVariableOp_228
lstm_cell/ReadVariableOp_3lstm_cell/ReadVariableOp_32@
lstm_cell/split/ReadVariableOplstm_cell/split/ReadVariableOp2D
 lstm_cell/split_1/ReadVariableOp lstm_cell/split_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:???????????????????
"
_user_specified_name
inputs/0
?
?
+__inference_Output_lay_layer_call_fn_121980

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????`*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_Output_lay_layer_call_and_return_conditional_losses_1200382
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????`2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?V
?
E__inference_lstm_cell_layer_call_and_return_conditional_losses_122071

inputs
states_0
states_1!
split_readvariableop_resource#
split_1_readvariableop_resource
readvariableop_resource
identity

identity_1

identity_2??ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?ReadVariableOp_3?split/ReadVariableOp?split_1/ReadVariableOpP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource* 
_output_shapes
:
??*
dtype02
split/ReadVariableOp?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
??:
??:
??:
??*
	num_split2
splite
MatMulMatMulinputssplit:output:0*
T0*(
_output_shapes
:??????????2
MatMuli
MatMul_1MatMulinputssplit:output:1*
T0*(
_output_shapes
:??????????2

MatMul_1i
MatMul_2MatMulinputssplit:output:2*
T0*(
_output_shapes
:??????????2

MatMul_2i
MatMul_3MatMulinputssplit:output:3*
T0*(
_output_shapes
:??????????2

MatMul_3T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_1/split_dim?
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:?*
dtype02
split_1/ReadVariableOp?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2	
split_1t
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*(
_output_shapes
:??????????2	
BiasAddz
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*(
_output_shapes
:??????????2
	BiasAdd_1z
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*(
_output_shapes
:??????????2
	BiasAdd_2z
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*(
_output_shapes
:??????????2
	BiasAdd_3z
ReadVariableOpReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype02
ReadVariableOp{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
strided_slices
MatMul_4MatMulstates_0strided_slice:output:0*
T0*(
_output_shapes
:??????????2

MatMul_4l
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
addW
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_2W
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_3_
MulMuladd:z:0Const_2:output:0*
T0*(
_output_shapes
:??????????2
Mulc
Add_1AddMul:z:0Const_3:output:0*
T0*(
_output_shapes
:??????????2
Add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:??????????2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*(
_output_shapes
:??????????2
clip_by_value~
ReadVariableOp_1ReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype02
ReadVariableOp_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
strided_slice_1u
MatMul_5MatMulstates_0strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2

MatMul_5r
add_2AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
add_2W
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_4W
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_5e
Mul_1Mul	add_2:z:0Const_4:output:0*
T0*(
_output_shapes
:??????????2
Mul_1e
Add_3Add	Mul_1:z:0Const_5:output:0*
T0*(
_output_shapes
:??????????2
Add_3{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_1/Minimum/y?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:??????????2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*(
_output_shapes
:??????????2
clip_by_value_1g
mul_2Mulclip_by_value_1:z:0states_1*
T0*(
_output_shapes
:??????????2
mul_2~
ReadVariableOp_2ReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype02
ReadVariableOp_2
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2
strided_slice_2/stack_1?
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2?
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
strided_slice_2u
MatMul_6MatMulstates_0strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2

MatMul_6r
add_4AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
add_4[
SigmoidSigmoid	add_4:z:0*
T0*(
_output_shapes
:??????????2	
Sigmoidh
mul_3Mulclip_by_value:z:0Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
mul_3`
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*(
_output_shapes
:??????????2
add_5~
ReadVariableOp_3ReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype02
ReadVariableOp_3
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2
strided_slice_3/stack?
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack_1?
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2?
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
strided_slice_3u
MatMul_7MatMulstates_0strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2

MatMul_7r
add_6AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
add_6W
Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_6W
Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_7e
Mul_4Mul	add_6:z:0Const_6:output:0*
T0*(
_output_shapes
:??????????2
Mul_4e
Add_7Add	Mul_4:z:0Const_7:output:0*
T0*(
_output_shapes
:??????????2
Add_7{
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_2/Minimum/y?
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:??????????2
clip_by_value_2/Minimumk
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_2/y?
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*(
_output_shapes
:??????????2
clip_by_value_2_
	Sigmoid_1Sigmoid	add_5:z:0*
T0*(
_output_shapes
:??????????2
	Sigmoid_1l
mul_5Mulclip_by_value_2:z:0Sigmoid_1:y:0*
T0*(
_output_shapes
:??????????2
mul_5?
IdentityIdentity	mul_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity	mul_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity_1?

Identity_2Identity	add_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*[
_input_shapesJ
H:??????????:??????????:??????????:::2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/1
??
?
LSTM_lay_0_while_body_1206042
.lstm_lay_0_while_lstm_lay_0_while_loop_counter8
4lstm_lay_0_while_lstm_lay_0_while_maximum_iterations 
lstm_lay_0_while_placeholder"
lstm_lay_0_while_placeholder_1"
lstm_lay_0_while_placeholder_2"
lstm_lay_0_while_placeholder_31
-lstm_lay_0_while_lstm_lay_0_strided_slice_1_0m
ilstm_lay_0_while_tensorarrayv2read_tensorlistgetitem_lstm_lay_0_tensorarrayunstack_tensorlistfromtensor_0>
:lstm_lay_0_while_lstm_cell_split_readvariableop_resource_0@
<lstm_lay_0_while_lstm_cell_split_1_readvariableop_resource_08
4lstm_lay_0_while_lstm_cell_readvariableop_resource_0
lstm_lay_0_while_identity
lstm_lay_0_while_identity_1
lstm_lay_0_while_identity_2
lstm_lay_0_while_identity_3
lstm_lay_0_while_identity_4
lstm_lay_0_while_identity_5/
+lstm_lay_0_while_lstm_lay_0_strided_slice_1k
glstm_lay_0_while_tensorarrayv2read_tensorlistgetitem_lstm_lay_0_tensorarrayunstack_tensorlistfromtensor<
8lstm_lay_0_while_lstm_cell_split_readvariableop_resource>
:lstm_lay_0_while_lstm_cell_split_1_readvariableop_resource6
2lstm_lay_0_while_lstm_cell_readvariableop_resource??)LSTM_lay_0/while/lstm_cell/ReadVariableOp?+LSTM_lay_0/while/lstm_cell/ReadVariableOp_1?+LSTM_lay_0/while/lstm_cell/ReadVariableOp_2?+LSTM_lay_0/while/lstm_cell/ReadVariableOp_3?/LSTM_lay_0/while/lstm_cell/split/ReadVariableOp?1LSTM_lay_0/while/lstm_cell/split_1/ReadVariableOp?
BLSTM_lay_0/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2D
BLSTM_lay_0/while/TensorArrayV2Read/TensorListGetItem/element_shape?
4LSTM_lay_0/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemilstm_lay_0_while_tensorarrayv2read_tensorlistgetitem_lstm_lay_0_tensorarrayunstack_tensorlistfromtensor_0lstm_lay_0_while_placeholderKLSTM_lay_0/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype026
4LSTM_lay_0/while/TensorArrayV2Read/TensorListGetItem?
 LSTM_lay_0/while/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2"
 LSTM_lay_0/while/lstm_cell/Const?
*LSTM_lay_0/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*LSTM_lay_0/while/lstm_cell/split/split_dim?
/LSTM_lay_0/while/lstm_cell/split/ReadVariableOpReadVariableOp:lstm_lay_0_while_lstm_cell_split_readvariableop_resource_0* 
_output_shapes
:
??*
dtype021
/LSTM_lay_0/while/lstm_cell/split/ReadVariableOp?
 LSTM_lay_0/while/lstm_cell/splitSplit3LSTM_lay_0/while/lstm_cell/split/split_dim:output:07LSTM_lay_0/while/lstm_cell/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
??:
??:
??:
??*
	num_split2"
 LSTM_lay_0/while/lstm_cell/split?
!LSTM_lay_0/while/lstm_cell/MatMulMatMul;LSTM_lay_0/while/TensorArrayV2Read/TensorListGetItem:item:0)LSTM_lay_0/while/lstm_cell/split:output:0*
T0*(
_output_shapes
:??????????2#
!LSTM_lay_0/while/lstm_cell/MatMul?
#LSTM_lay_0/while/lstm_cell/MatMul_1MatMul;LSTM_lay_0/while/TensorArrayV2Read/TensorListGetItem:item:0)LSTM_lay_0/while/lstm_cell/split:output:1*
T0*(
_output_shapes
:??????????2%
#LSTM_lay_0/while/lstm_cell/MatMul_1?
#LSTM_lay_0/while/lstm_cell/MatMul_2MatMul;LSTM_lay_0/while/TensorArrayV2Read/TensorListGetItem:item:0)LSTM_lay_0/while/lstm_cell/split:output:2*
T0*(
_output_shapes
:??????????2%
#LSTM_lay_0/while/lstm_cell/MatMul_2?
#LSTM_lay_0/while/lstm_cell/MatMul_3MatMul;LSTM_lay_0/while/TensorArrayV2Read/TensorListGetItem:item:0)LSTM_lay_0/while/lstm_cell/split:output:3*
T0*(
_output_shapes
:??????????2%
#LSTM_lay_0/while/lstm_cell/MatMul_3?
"LSTM_lay_0/while/lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2$
"LSTM_lay_0/while/lstm_cell/Const_1?
,LSTM_lay_0/while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2.
,LSTM_lay_0/while/lstm_cell/split_1/split_dim?
1LSTM_lay_0/while/lstm_cell/split_1/ReadVariableOpReadVariableOp<lstm_lay_0_while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
:?*
dtype023
1LSTM_lay_0/while/lstm_cell/split_1/ReadVariableOp?
"LSTM_lay_0/while/lstm_cell/split_1Split5LSTM_lay_0/while/lstm_cell/split_1/split_dim:output:09LSTM_lay_0/while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2$
"LSTM_lay_0/while/lstm_cell/split_1?
"LSTM_lay_0/while/lstm_cell/BiasAddBiasAdd+LSTM_lay_0/while/lstm_cell/MatMul:product:0+LSTM_lay_0/while/lstm_cell/split_1:output:0*
T0*(
_output_shapes
:??????????2$
"LSTM_lay_0/while/lstm_cell/BiasAdd?
$LSTM_lay_0/while/lstm_cell/BiasAdd_1BiasAdd-LSTM_lay_0/while/lstm_cell/MatMul_1:product:0+LSTM_lay_0/while/lstm_cell/split_1:output:1*
T0*(
_output_shapes
:??????????2&
$LSTM_lay_0/while/lstm_cell/BiasAdd_1?
$LSTM_lay_0/while/lstm_cell/BiasAdd_2BiasAdd-LSTM_lay_0/while/lstm_cell/MatMul_2:product:0+LSTM_lay_0/while/lstm_cell/split_1:output:2*
T0*(
_output_shapes
:??????????2&
$LSTM_lay_0/while/lstm_cell/BiasAdd_2?
$LSTM_lay_0/while/lstm_cell/BiasAdd_3BiasAdd-LSTM_lay_0/while/lstm_cell/MatMul_3:product:0+LSTM_lay_0/while/lstm_cell/split_1:output:3*
T0*(
_output_shapes
:??????????2&
$LSTM_lay_0/while/lstm_cell/BiasAdd_3?
)LSTM_lay_0/while/lstm_cell/ReadVariableOpReadVariableOp4lstm_lay_0_while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02+
)LSTM_lay_0/while/lstm_cell/ReadVariableOp?
.LSTM_lay_0/while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        20
.LSTM_lay_0/while/lstm_cell/strided_slice/stack?
0LSTM_lay_0/while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   22
0LSTM_lay_0/while/lstm_cell/strided_slice/stack_1?
0LSTM_lay_0/while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0LSTM_lay_0/while/lstm_cell/strided_slice/stack_2?
(LSTM_lay_0/while/lstm_cell/strided_sliceStridedSlice1LSTM_lay_0/while/lstm_cell/ReadVariableOp:value:07LSTM_lay_0/while/lstm_cell/strided_slice/stack:output:09LSTM_lay_0/while/lstm_cell/strided_slice/stack_1:output:09LSTM_lay_0/while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2*
(LSTM_lay_0/while/lstm_cell/strided_slice?
#LSTM_lay_0/while/lstm_cell/MatMul_4MatMullstm_lay_0_while_placeholder_21LSTM_lay_0/while/lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:??????????2%
#LSTM_lay_0/while/lstm_cell/MatMul_4?
LSTM_lay_0/while/lstm_cell/addAddV2+LSTM_lay_0/while/lstm_cell/BiasAdd:output:0-LSTM_lay_0/while/lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2 
LSTM_lay_0/while/lstm_cell/add?
"LSTM_lay_0/while/lstm_cell/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2$
"LSTM_lay_0/while/lstm_cell/Const_2?
"LSTM_lay_0/while/lstm_cell/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2$
"LSTM_lay_0/while/lstm_cell/Const_3?
LSTM_lay_0/while/lstm_cell/MulMul"LSTM_lay_0/while/lstm_cell/add:z:0+LSTM_lay_0/while/lstm_cell/Const_2:output:0*
T0*(
_output_shapes
:??????????2 
LSTM_lay_0/while/lstm_cell/Mul?
 LSTM_lay_0/while/lstm_cell/Add_1Add"LSTM_lay_0/while/lstm_cell/Mul:z:0+LSTM_lay_0/while/lstm_cell/Const_3:output:0*
T0*(
_output_shapes
:??????????2"
 LSTM_lay_0/while/lstm_cell/Add_1?
2LSTM_lay_0/while/lstm_cell/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??24
2LSTM_lay_0/while/lstm_cell/clip_by_value/Minimum/y?
0LSTM_lay_0/while/lstm_cell/clip_by_value/MinimumMinimum$LSTM_lay_0/while/lstm_cell/Add_1:z:0;LSTM_lay_0/while/lstm_cell/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:??????????22
0LSTM_lay_0/while/lstm_cell/clip_by_value/Minimum?
*LSTM_lay_0/while/lstm_cell/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*LSTM_lay_0/while/lstm_cell/clip_by_value/y?
(LSTM_lay_0/while/lstm_cell/clip_by_valueMaximum4LSTM_lay_0/while/lstm_cell/clip_by_value/Minimum:z:03LSTM_lay_0/while/lstm_cell/clip_by_value/y:output:0*
T0*(
_output_shapes
:??????????2*
(LSTM_lay_0/while/lstm_cell/clip_by_value?
+LSTM_lay_0/while/lstm_cell/ReadVariableOp_1ReadVariableOp4lstm_lay_0_while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02-
+LSTM_lay_0/while/lstm_cell/ReadVariableOp_1?
0LSTM_lay_0/while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   22
0LSTM_lay_0/while/lstm_cell/strided_slice_1/stack?
2LSTM_lay_0/while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       24
2LSTM_lay_0/while/lstm_cell/strided_slice_1/stack_1?
2LSTM_lay_0/while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      24
2LSTM_lay_0/while/lstm_cell/strided_slice_1/stack_2?
*LSTM_lay_0/while/lstm_cell/strided_slice_1StridedSlice3LSTM_lay_0/while/lstm_cell/ReadVariableOp_1:value:09LSTM_lay_0/while/lstm_cell/strided_slice_1/stack:output:0;LSTM_lay_0/while/lstm_cell/strided_slice_1/stack_1:output:0;LSTM_lay_0/while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2,
*LSTM_lay_0/while/lstm_cell/strided_slice_1?
#LSTM_lay_0/while/lstm_cell/MatMul_5MatMullstm_lay_0_while_placeholder_23LSTM_lay_0/while/lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2%
#LSTM_lay_0/while/lstm_cell/MatMul_5?
 LSTM_lay_0/while/lstm_cell/add_2AddV2-LSTM_lay_0/while/lstm_cell/BiasAdd_1:output:0-LSTM_lay_0/while/lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2"
 LSTM_lay_0/while/lstm_cell/add_2?
"LSTM_lay_0/while/lstm_cell/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2$
"LSTM_lay_0/while/lstm_cell/Const_4?
"LSTM_lay_0/while/lstm_cell/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2$
"LSTM_lay_0/while/lstm_cell/Const_5?
 LSTM_lay_0/while/lstm_cell/Mul_1Mul$LSTM_lay_0/while/lstm_cell/add_2:z:0+LSTM_lay_0/while/lstm_cell/Const_4:output:0*
T0*(
_output_shapes
:??????????2"
 LSTM_lay_0/while/lstm_cell/Mul_1?
 LSTM_lay_0/while/lstm_cell/Add_3Add$LSTM_lay_0/while/lstm_cell/Mul_1:z:0+LSTM_lay_0/while/lstm_cell/Const_5:output:0*
T0*(
_output_shapes
:??????????2"
 LSTM_lay_0/while/lstm_cell/Add_3?
4LSTM_lay_0/while/lstm_cell/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??26
4LSTM_lay_0/while/lstm_cell/clip_by_value_1/Minimum/y?
2LSTM_lay_0/while/lstm_cell/clip_by_value_1/MinimumMinimum$LSTM_lay_0/while/lstm_cell/Add_3:z:0=LSTM_lay_0/while/lstm_cell/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:??????????24
2LSTM_lay_0/while/lstm_cell/clip_by_value_1/Minimum?
,LSTM_lay_0/while/lstm_cell/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2.
,LSTM_lay_0/while/lstm_cell/clip_by_value_1/y?
*LSTM_lay_0/while/lstm_cell/clip_by_value_1Maximum6LSTM_lay_0/while/lstm_cell/clip_by_value_1/Minimum:z:05LSTM_lay_0/while/lstm_cell/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:??????????2,
*LSTM_lay_0/while/lstm_cell/clip_by_value_1?
 LSTM_lay_0/while/lstm_cell/mul_2Mul.LSTM_lay_0/while/lstm_cell/clip_by_value_1:z:0lstm_lay_0_while_placeholder_3*
T0*(
_output_shapes
:??????????2"
 LSTM_lay_0/while/lstm_cell/mul_2?
+LSTM_lay_0/while/lstm_cell/ReadVariableOp_2ReadVariableOp4lstm_lay_0_while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02-
+LSTM_lay_0/while/lstm_cell/ReadVariableOp_2?
0LSTM_lay_0/while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       22
0LSTM_lay_0/while/lstm_cell/strided_slice_2/stack?
2LSTM_lay_0/while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  24
2LSTM_lay_0/while/lstm_cell/strided_slice_2/stack_1?
2LSTM_lay_0/while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      24
2LSTM_lay_0/while/lstm_cell/strided_slice_2/stack_2?
*LSTM_lay_0/while/lstm_cell/strided_slice_2StridedSlice3LSTM_lay_0/while/lstm_cell/ReadVariableOp_2:value:09LSTM_lay_0/while/lstm_cell/strided_slice_2/stack:output:0;LSTM_lay_0/while/lstm_cell/strided_slice_2/stack_1:output:0;LSTM_lay_0/while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2,
*LSTM_lay_0/while/lstm_cell/strided_slice_2?
#LSTM_lay_0/while/lstm_cell/MatMul_6MatMullstm_lay_0_while_placeholder_23LSTM_lay_0/while/lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2%
#LSTM_lay_0/while/lstm_cell/MatMul_6?
 LSTM_lay_0/while/lstm_cell/add_4AddV2-LSTM_lay_0/while/lstm_cell/BiasAdd_2:output:0-LSTM_lay_0/while/lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2"
 LSTM_lay_0/while/lstm_cell/add_4?
"LSTM_lay_0/while/lstm_cell/SigmoidSigmoid$LSTM_lay_0/while/lstm_cell/add_4:z:0*
T0*(
_output_shapes
:??????????2$
"LSTM_lay_0/while/lstm_cell/Sigmoid?
 LSTM_lay_0/while/lstm_cell/mul_3Mul,LSTM_lay_0/while/lstm_cell/clip_by_value:z:0&LSTM_lay_0/while/lstm_cell/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2"
 LSTM_lay_0/while/lstm_cell/mul_3?
 LSTM_lay_0/while/lstm_cell/add_5AddV2$LSTM_lay_0/while/lstm_cell/mul_2:z:0$LSTM_lay_0/while/lstm_cell/mul_3:z:0*
T0*(
_output_shapes
:??????????2"
 LSTM_lay_0/while/lstm_cell/add_5?
+LSTM_lay_0/while/lstm_cell/ReadVariableOp_3ReadVariableOp4lstm_lay_0_while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02-
+LSTM_lay_0/while/lstm_cell/ReadVariableOp_3?
0LSTM_lay_0/while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  22
0LSTM_lay_0/while/lstm_cell/strided_slice_3/stack?
2LSTM_lay_0/while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        24
2LSTM_lay_0/while/lstm_cell/strided_slice_3/stack_1?
2LSTM_lay_0/while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      24
2LSTM_lay_0/while/lstm_cell/strided_slice_3/stack_2?
*LSTM_lay_0/while/lstm_cell/strided_slice_3StridedSlice3LSTM_lay_0/while/lstm_cell/ReadVariableOp_3:value:09LSTM_lay_0/while/lstm_cell/strided_slice_3/stack:output:0;LSTM_lay_0/while/lstm_cell/strided_slice_3/stack_1:output:0;LSTM_lay_0/while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2,
*LSTM_lay_0/while/lstm_cell/strided_slice_3?
#LSTM_lay_0/while/lstm_cell/MatMul_7MatMullstm_lay_0_while_placeholder_23LSTM_lay_0/while/lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2%
#LSTM_lay_0/while/lstm_cell/MatMul_7?
 LSTM_lay_0/while/lstm_cell/add_6AddV2-LSTM_lay_0/while/lstm_cell/BiasAdd_3:output:0-LSTM_lay_0/while/lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2"
 LSTM_lay_0/while/lstm_cell/add_6?
"LSTM_lay_0/while/lstm_cell/Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2$
"LSTM_lay_0/while/lstm_cell/Const_6?
"LSTM_lay_0/while/lstm_cell/Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2$
"LSTM_lay_0/while/lstm_cell/Const_7?
 LSTM_lay_0/while/lstm_cell/Mul_4Mul$LSTM_lay_0/while/lstm_cell/add_6:z:0+LSTM_lay_0/while/lstm_cell/Const_6:output:0*
T0*(
_output_shapes
:??????????2"
 LSTM_lay_0/while/lstm_cell/Mul_4?
 LSTM_lay_0/while/lstm_cell/Add_7Add$LSTM_lay_0/while/lstm_cell/Mul_4:z:0+LSTM_lay_0/while/lstm_cell/Const_7:output:0*
T0*(
_output_shapes
:??????????2"
 LSTM_lay_0/while/lstm_cell/Add_7?
4LSTM_lay_0/while/lstm_cell/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??26
4LSTM_lay_0/while/lstm_cell/clip_by_value_2/Minimum/y?
2LSTM_lay_0/while/lstm_cell/clip_by_value_2/MinimumMinimum$LSTM_lay_0/while/lstm_cell/Add_7:z:0=LSTM_lay_0/while/lstm_cell/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:??????????24
2LSTM_lay_0/while/lstm_cell/clip_by_value_2/Minimum?
,LSTM_lay_0/while/lstm_cell/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2.
,LSTM_lay_0/while/lstm_cell/clip_by_value_2/y?
*LSTM_lay_0/while/lstm_cell/clip_by_value_2Maximum6LSTM_lay_0/while/lstm_cell/clip_by_value_2/Minimum:z:05LSTM_lay_0/while/lstm_cell/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:??????????2,
*LSTM_lay_0/while/lstm_cell/clip_by_value_2?
$LSTM_lay_0/while/lstm_cell/Sigmoid_1Sigmoid$LSTM_lay_0/while/lstm_cell/add_5:z:0*
T0*(
_output_shapes
:??????????2&
$LSTM_lay_0/while/lstm_cell/Sigmoid_1?
 LSTM_lay_0/while/lstm_cell/mul_5Mul.LSTM_lay_0/while/lstm_cell/clip_by_value_2:z:0(LSTM_lay_0/while/lstm_cell/Sigmoid_1:y:0*
T0*(
_output_shapes
:??????????2"
 LSTM_lay_0/while/lstm_cell/mul_5?
5LSTM_lay_0/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_lay_0_while_placeholder_1lstm_lay_0_while_placeholder$LSTM_lay_0/while/lstm_cell/mul_5:z:0*
_output_shapes
: *
element_dtype027
5LSTM_lay_0/while/TensorArrayV2Write/TensorListSetItemr
LSTM_lay_0/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
LSTM_lay_0/while/add/y?
LSTM_lay_0/while/addAddV2lstm_lay_0_while_placeholderLSTM_lay_0/while/add/y:output:0*
T0*
_output_shapes
: 2
LSTM_lay_0/while/addv
LSTM_lay_0/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
LSTM_lay_0/while/add_1/y?
LSTM_lay_0/while/add_1AddV2.lstm_lay_0_while_lstm_lay_0_while_loop_counter!LSTM_lay_0/while/add_1/y:output:0*
T0*
_output_shapes
: 2
LSTM_lay_0/while/add_1?
LSTM_lay_0/while/IdentityIdentityLSTM_lay_0/while/add_1:z:0*^LSTM_lay_0/while/lstm_cell/ReadVariableOp,^LSTM_lay_0/while/lstm_cell/ReadVariableOp_1,^LSTM_lay_0/while/lstm_cell/ReadVariableOp_2,^LSTM_lay_0/while/lstm_cell/ReadVariableOp_30^LSTM_lay_0/while/lstm_cell/split/ReadVariableOp2^LSTM_lay_0/while/lstm_cell/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
LSTM_lay_0/while/Identity?
LSTM_lay_0/while/Identity_1Identity4lstm_lay_0_while_lstm_lay_0_while_maximum_iterations*^LSTM_lay_0/while/lstm_cell/ReadVariableOp,^LSTM_lay_0/while/lstm_cell/ReadVariableOp_1,^LSTM_lay_0/while/lstm_cell/ReadVariableOp_2,^LSTM_lay_0/while/lstm_cell/ReadVariableOp_30^LSTM_lay_0/while/lstm_cell/split/ReadVariableOp2^LSTM_lay_0/while/lstm_cell/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
LSTM_lay_0/while/Identity_1?
LSTM_lay_0/while/Identity_2IdentityLSTM_lay_0/while/add:z:0*^LSTM_lay_0/while/lstm_cell/ReadVariableOp,^LSTM_lay_0/while/lstm_cell/ReadVariableOp_1,^LSTM_lay_0/while/lstm_cell/ReadVariableOp_2,^LSTM_lay_0/while/lstm_cell/ReadVariableOp_30^LSTM_lay_0/while/lstm_cell/split/ReadVariableOp2^LSTM_lay_0/while/lstm_cell/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
LSTM_lay_0/while/Identity_2?
LSTM_lay_0/while/Identity_3IdentityELSTM_lay_0/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^LSTM_lay_0/while/lstm_cell/ReadVariableOp,^LSTM_lay_0/while/lstm_cell/ReadVariableOp_1,^LSTM_lay_0/while/lstm_cell/ReadVariableOp_2,^LSTM_lay_0/while/lstm_cell/ReadVariableOp_30^LSTM_lay_0/while/lstm_cell/split/ReadVariableOp2^LSTM_lay_0/while/lstm_cell/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
LSTM_lay_0/while/Identity_3?
LSTM_lay_0/while/Identity_4Identity$LSTM_lay_0/while/lstm_cell/mul_5:z:0*^LSTM_lay_0/while/lstm_cell/ReadVariableOp,^LSTM_lay_0/while/lstm_cell/ReadVariableOp_1,^LSTM_lay_0/while/lstm_cell/ReadVariableOp_2,^LSTM_lay_0/while/lstm_cell/ReadVariableOp_30^LSTM_lay_0/while/lstm_cell/split/ReadVariableOp2^LSTM_lay_0/while/lstm_cell/split_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
LSTM_lay_0/while/Identity_4?
LSTM_lay_0/while/Identity_5Identity$LSTM_lay_0/while/lstm_cell/add_5:z:0*^LSTM_lay_0/while/lstm_cell/ReadVariableOp,^LSTM_lay_0/while/lstm_cell/ReadVariableOp_1,^LSTM_lay_0/while/lstm_cell/ReadVariableOp_2,^LSTM_lay_0/while/lstm_cell/ReadVariableOp_30^LSTM_lay_0/while/lstm_cell/split/ReadVariableOp2^LSTM_lay_0/while/lstm_cell/split_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
LSTM_lay_0/while/Identity_5"?
lstm_lay_0_while_identity"LSTM_lay_0/while/Identity:output:0"C
lstm_lay_0_while_identity_1$LSTM_lay_0/while/Identity_1:output:0"C
lstm_lay_0_while_identity_2$LSTM_lay_0/while/Identity_2:output:0"C
lstm_lay_0_while_identity_3$LSTM_lay_0/while/Identity_3:output:0"C
lstm_lay_0_while_identity_4$LSTM_lay_0/while/Identity_4:output:0"C
lstm_lay_0_while_identity_5$LSTM_lay_0/while/Identity_5:output:0"j
2lstm_lay_0_while_lstm_cell_readvariableop_resource4lstm_lay_0_while_lstm_cell_readvariableop_resource_0"z
:lstm_lay_0_while_lstm_cell_split_1_readvariableop_resource<lstm_lay_0_while_lstm_cell_split_1_readvariableop_resource_0"v
8lstm_lay_0_while_lstm_cell_split_readvariableop_resource:lstm_lay_0_while_lstm_cell_split_readvariableop_resource_0"\
+lstm_lay_0_while_lstm_lay_0_strided_slice_1-lstm_lay_0_while_lstm_lay_0_strided_slice_1_0"?
glstm_lay_0_while_tensorarrayv2read_tensorlistgetitem_lstm_lay_0_tensorarrayunstack_tensorlistfromtensorilstm_lay_0_while_tensorarrayv2read_tensorlistgetitem_lstm_lay_0_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :??????????:??????????: : :::2V
)LSTM_lay_0/while/lstm_cell/ReadVariableOp)LSTM_lay_0/while/lstm_cell/ReadVariableOp2Z
+LSTM_lay_0/while/lstm_cell/ReadVariableOp_1+LSTM_lay_0/while/lstm_cell/ReadVariableOp_12Z
+LSTM_lay_0/while/lstm_cell/ReadVariableOp_2+LSTM_lay_0/while/lstm_cell/ReadVariableOp_22Z
+LSTM_lay_0/while/lstm_cell/ReadVariableOp_3+LSTM_lay_0/while/lstm_cell/ReadVariableOp_32b
/LSTM_lay_0/while/lstm_cell/split/ReadVariableOp/LSTM_lay_0/while/lstm_cell/split/ReadVariableOp2f
1LSTM_lay_0/while/lstm_cell/split_1/ReadVariableOp1LSTM_lay_0/while/lstm_cell/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
??
?
F__inference_LSTM_lay_0_layer_call_and_return_conditional_losses_121334

inputs+
'lstm_cell_split_readvariableop_resource-
)lstm_cell_split_1_readvariableop_resource%
!lstm_cell_readvariableop_resource
identity??lstm_cell/ReadVariableOp?lstm_cell/ReadVariableOp_1?lstm_cell/ReadVariableOp_2?lstm_cell/ReadVariableOp_3?lstm_cell/split/ReadVariableOp? lstm_cell/split_1/ReadVariableOp?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:??????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_2d
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dim?
lstm_cell/split/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
lstm_cell/split/ReadVariableOp?
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
??:
??:
??:
??*
	num_split2
lstm_cell/split?
lstm_cell/MatMulMatMulstrided_slice_2:output:0lstm_cell/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul?
lstm_cell/MatMul_1MatMulstrided_slice_2:output:0lstm_cell/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_1?
lstm_cell/MatMul_2MatMulstrided_slice_2:output:0lstm_cell/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_2?
lstm_cell/MatMul_3MatMulstrided_slice_2:output:0lstm_cell/split:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_3h
lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Const_1|
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell/split_1/split_dim?
 lstm_cell/split_1/ReadVariableOpReadVariableOp)lstm_cell_split_1_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 lstm_cell/split_1/ReadVariableOp?
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2
lstm_cell/split_1?
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd?
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_1?
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_2?
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_3?
lstm_cell/ReadVariableOpReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp?
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
lstm_cell/strided_slice/stack?
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2!
lstm_cell/strided_slice/stack_1?
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
lstm_cell/strided_slice/stack_2?
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice?
lstm_cell/MatMul_4MatMulzeros:output:0 lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_4?
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/addk
lstm_cell/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
lstm_cell/Const_2k
lstm_cell/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm_cell/Const_3?
lstm_cell/MulMullstm_cell/add:z:0lstm_cell/Const_2:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Mul?
lstm_cell/Add_1Addlstm_cell/Mul:z:0lstm_cell/Const_3:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Add_1?
!lstm_cell/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2#
!lstm_cell/clip_by_value/Minimum/y?
lstm_cell/clip_by_value/MinimumMinimumlstm_cell/Add_1:z:0*lstm_cell/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:??????????2!
lstm_cell/clip_by_value/Minimum{
lstm_cell/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_cell/clip_by_value/y?
lstm_cell/clip_by_valueMaximum#lstm_cell/clip_by_value/Minimum:z:0"lstm_cell/clip_by_value/y:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/clip_by_value?
lstm_cell/ReadVariableOp_1ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_1?
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2!
lstm_cell/strided_slice_1/stack?
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_1/stack_1?
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_1/stack_2?
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_1?
lstm_cell/MatMul_5MatMulzeros:output:0"lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_5?
lstm_cell/add_2AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_2k
lstm_cell/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
lstm_cell/Const_4k
lstm_cell/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm_cell/Const_5?
lstm_cell/Mul_1Mullstm_cell/add_2:z:0lstm_cell/Const_4:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Mul_1?
lstm_cell/Add_3Addlstm_cell/Mul_1:z:0lstm_cell/Const_5:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Add_3?
#lstm_cell/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2%
#lstm_cell/clip_by_value_1/Minimum/y?
!lstm_cell/clip_by_value_1/MinimumMinimumlstm_cell/Add_3:z:0,lstm_cell/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:??????????2#
!lstm_cell/clip_by_value_1/Minimum
lstm_cell/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_cell/clip_by_value_1/y?
lstm_cell/clip_by_value_1Maximum%lstm_cell/clip_by_value_1/Minimum:z:0$lstm_cell/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/clip_by_value_1?
lstm_cell/mul_2Mullstm_cell/clip_by_value_1:z:0zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_2?
lstm_cell/ReadVariableOp_2ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_2?
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_2/stack?
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2#
!lstm_cell/strided_slice_2/stack_1?
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_2/stack_2?
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_2?
lstm_cell/MatMul_6MatMulzeros:output:0"lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_6?
lstm_cell/add_4AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_4y
lstm_cell/SigmoidSigmoidlstm_cell/add_4:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid?
lstm_cell/mul_3Mullstm_cell/clip_by_value:z:0lstm_cell/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_3?
lstm_cell/add_5AddV2lstm_cell/mul_2:z:0lstm_cell/mul_3:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_5?
lstm_cell/ReadVariableOp_3ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_3?
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2!
lstm_cell/strided_slice_3/stack?
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_3/stack_1?
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_3/stack_2?
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_3?
lstm_cell/MatMul_7MatMulzeros:output:0"lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_7?
lstm_cell/add_6AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_6k
lstm_cell/Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
lstm_cell/Const_6k
lstm_cell/Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm_cell/Const_7?
lstm_cell/Mul_4Mullstm_cell/add_6:z:0lstm_cell/Const_6:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Mul_4?
lstm_cell/Add_7Addlstm_cell/Mul_4:z:0lstm_cell/Const_7:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Add_7?
#lstm_cell/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2%
#lstm_cell/clip_by_value_2/Minimum/y?
!lstm_cell/clip_by_value_2/MinimumMinimumlstm_cell/Add_7:z:0,lstm_cell/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:??????????2#
!lstm_cell/clip_by_value_2/Minimum
lstm_cell/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_cell/clip_by_value_2/y?
lstm_cell/clip_by_value_2Maximum%lstm_cell/clip_by_value_2/Minimum:z:0$lstm_cell/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/clip_by_value_2}
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_5:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid_1?
lstm_cell/mul_5Mullstm_cell/clip_by_value_2:z:0lstm_cell/Sigmoid_1:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_5?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0'lstm_cell_split_readvariableop_resource)lstm_cell_split_1_readvariableop_resource!lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_121192*
condR
while_cond_121191*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2
transpose_1?
IdentityIdentitystrided_slice_3:output:0^lstm_cell/ReadVariableOp^lstm_cell/ReadVariableOp_1^lstm_cell/ReadVariableOp_2^lstm_cell/ReadVariableOp_3^lstm_cell/split/ReadVariableOp!^lstm_cell/split_1/ReadVariableOp^while*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????:::24
lstm_cell/ReadVariableOplstm_cell/ReadVariableOp28
lstm_cell/ReadVariableOp_1lstm_cell/ReadVariableOp_128
lstm_cell/ReadVariableOp_2lstm_cell/ReadVariableOp_228
lstm_cell/ReadVariableOp_3lstm_cell/ReadVariableOp_32@
lstm_cell/split/ReadVariableOplstm_cell/split/ReadVariableOp2D
 lstm_cell/split_1/ReadVariableOp lstm_cell/split_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
F__inference_Output_lay_layer_call_and_return_conditional_losses_121971

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?`*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????`2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:`*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????`2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????`2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
while_body_119799
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_03
/while_lstm_cell_split_readvariableop_resource_05
1while_lstm_cell_split_1_readvariableop_resource_0-
)while_lstm_cell_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor1
-while_lstm_cell_split_readvariableop_resource3
/while_lstm_cell_split_1_readvariableop_resource+
'while_lstm_cell_readvariableop_resource??while/lstm_cell/ReadVariableOp? while/lstm_cell/ReadVariableOp_1? while/lstm_cell/ReadVariableOp_2? while/lstm_cell/ReadVariableOp_3?$while/lstm_cell/split/ReadVariableOp?&while/lstm_cell/split_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemp
while/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell/Const?
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2!
while/lstm_cell/split/split_dim?
$while/lstm_cell/split/ReadVariableOpReadVariableOp/while_lstm_cell_split_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02&
$while/lstm_cell/split/ReadVariableOp?
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0,while/lstm_cell/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
??:
??:
??:
??*
	num_split2
while/lstm_cell/split?
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/MatMul?
while/lstm_cell/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:1*
T0*(
_output_shapes
:??????????2
while/lstm_cell/MatMul_1?
while/lstm_cell/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:2*
T0*(
_output_shapes
:??????????2
while/lstm_cell/MatMul_2?
while/lstm_cell/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:3*
T0*(
_output_shapes
:??????????2
while/lstm_cell/MatMul_3t
while/lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell/Const_1?
!while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!while/lstm_cell/split_1/split_dim?
&while/lstm_cell/split_1/ReadVariableOpReadVariableOp1while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
:?*
dtype02(
&while/lstm_cell/split_1/ReadVariableOp?
while/lstm_cell/split_1Split*while/lstm_cell/split_1/split_dim:output:0.while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2
while/lstm_cell/split_1?
while/lstm_cell/BiasAddBiasAdd while/lstm_cell/MatMul:product:0 while/lstm_cell/split_1:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/BiasAdd?
while/lstm_cell/BiasAdd_1BiasAdd"while/lstm_cell/MatMul_1:product:0 while/lstm_cell/split_1:output:1*
T0*(
_output_shapes
:??????????2
while/lstm_cell/BiasAdd_1?
while/lstm_cell/BiasAdd_2BiasAdd"while/lstm_cell/MatMul_2:product:0 while/lstm_cell/split_1:output:2*
T0*(
_output_shapes
:??????????2
while/lstm_cell/BiasAdd_2?
while/lstm_cell/BiasAdd_3BiasAdd"while/lstm_cell/MatMul_3:product:0 while/lstm_cell/split_1:output:3*
T0*(
_output_shapes
:??????????2
while/lstm_cell/BiasAdd_3?
while/lstm_cell/ReadVariableOpReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02 
while/lstm_cell/ReadVariableOp?
#while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2%
#while/lstm_cell/strided_slice/stack?
%while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2'
%while/lstm_cell/strided_slice/stack_1?
%while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2'
%while/lstm_cell/strided_slice/stack_2?
while/lstm_cell/strided_sliceStridedSlice&while/lstm_cell/ReadVariableOp:value:0,while/lstm_cell/strided_slice/stack:output:0.while/lstm_cell/strided_slice/stack_1:output:0.while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
while/lstm_cell/strided_slice?
while/lstm_cell/MatMul_4MatMulwhile_placeholder_2&while/lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/MatMul_4?
while/lstm_cell/addAddV2 while/lstm_cell/BiasAdd:output:0"while/lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/addw
while/lstm_cell/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/lstm_cell/Const_2w
while/lstm_cell/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/lstm_cell/Const_3?
while/lstm_cell/MulMulwhile/lstm_cell/add:z:0 while/lstm_cell/Const_2:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/Mul?
while/lstm_cell/Add_1Addwhile/lstm_cell/Mul:z:0 while/lstm_cell/Const_3:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/Add_1?
'while/lstm_cell/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2)
'while/lstm_cell/clip_by_value/Minimum/y?
%while/lstm_cell/clip_by_value/MinimumMinimumwhile/lstm_cell/Add_1:z:00while/lstm_cell/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:??????????2'
%while/lstm_cell/clip_by_value/Minimum?
while/lstm_cell/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
while/lstm_cell/clip_by_value/y?
while/lstm_cell/clip_by_valueMaximum)while/lstm_cell/clip_by_value/Minimum:z:0(while/lstm_cell/clip_by_value/y:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/clip_by_value?
 while/lstm_cell/ReadVariableOp_1ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02"
 while/lstm_cell/ReadVariableOp_1?
%while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2'
%while/lstm_cell/strided_slice_1/stack?
'while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2)
'while/lstm_cell/strided_slice_1/stack_1?
'while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_1/stack_2?
while/lstm_cell/strided_slice_1StridedSlice(while/lstm_cell/ReadVariableOp_1:value:0.while/lstm_cell/strided_slice_1/stack:output:00while/lstm_cell/strided_slice_1/stack_1:output:00while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2!
while/lstm_cell/strided_slice_1?
while/lstm_cell/MatMul_5MatMulwhile_placeholder_2(while/lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/MatMul_5?
while/lstm_cell/add_2AddV2"while/lstm_cell/BiasAdd_1:output:0"while/lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/add_2w
while/lstm_cell/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/lstm_cell/Const_4w
while/lstm_cell/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/lstm_cell/Const_5?
while/lstm_cell/Mul_1Mulwhile/lstm_cell/add_2:z:0 while/lstm_cell/Const_4:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/Mul_1?
while/lstm_cell/Add_3Addwhile/lstm_cell/Mul_1:z:0 while/lstm_cell/Const_5:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/Add_3?
)while/lstm_cell/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2+
)while/lstm_cell/clip_by_value_1/Minimum/y?
'while/lstm_cell/clip_by_value_1/MinimumMinimumwhile/lstm_cell/Add_3:z:02while/lstm_cell/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:??????????2)
'while/lstm_cell/clip_by_value_1/Minimum?
!while/lstm_cell/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!while/lstm_cell/clip_by_value_1/y?
while/lstm_cell/clip_by_value_1Maximum+while/lstm_cell/clip_by_value_1/Minimum:z:0*while/lstm_cell/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:??????????2!
while/lstm_cell/clip_by_value_1?
while/lstm_cell/mul_2Mul#while/lstm_cell/clip_by_value_1:z:0while_placeholder_3*
T0*(
_output_shapes
:??????????2
while/lstm_cell/mul_2?
 while/lstm_cell/ReadVariableOp_2ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02"
 while/lstm_cell/ReadVariableOp_2?
%while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2'
%while/lstm_cell/strided_slice_2/stack?
'while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2)
'while/lstm_cell/strided_slice_2/stack_1?
'while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_2/stack_2?
while/lstm_cell/strided_slice_2StridedSlice(while/lstm_cell/ReadVariableOp_2:value:0.while/lstm_cell/strided_slice_2/stack:output:00while/lstm_cell/strided_slice_2/stack_1:output:00while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2!
while/lstm_cell/strided_slice_2?
while/lstm_cell/MatMul_6MatMulwhile_placeholder_2(while/lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/MatMul_6?
while/lstm_cell/add_4AddV2"while/lstm_cell/BiasAdd_2:output:0"while/lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/add_4?
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/add_4:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/Sigmoid?
while/lstm_cell/mul_3Mul!while/lstm_cell/clip_by_value:z:0while/lstm_cell/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/mul_3?
while/lstm_cell/add_5AddV2while/lstm_cell/mul_2:z:0while/lstm_cell/mul_3:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/add_5?
 while/lstm_cell/ReadVariableOp_3ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02"
 while/lstm_cell/ReadVariableOp_3?
%while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2'
%while/lstm_cell/strided_slice_3/stack?
'while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'while/lstm_cell/strided_slice_3/stack_1?
'while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_3/stack_2?
while/lstm_cell/strided_slice_3StridedSlice(while/lstm_cell/ReadVariableOp_3:value:0.while/lstm_cell/strided_slice_3/stack:output:00while/lstm_cell/strided_slice_3/stack_1:output:00while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2!
while/lstm_cell/strided_slice_3?
while/lstm_cell/MatMul_7MatMulwhile_placeholder_2(while/lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/MatMul_7?
while/lstm_cell/add_6AddV2"while/lstm_cell/BiasAdd_3:output:0"while/lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/add_6w
while/lstm_cell/Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/lstm_cell/Const_6w
while/lstm_cell/Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/lstm_cell/Const_7?
while/lstm_cell/Mul_4Mulwhile/lstm_cell/add_6:z:0 while/lstm_cell/Const_6:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/Mul_4?
while/lstm_cell/Add_7Addwhile/lstm_cell/Mul_4:z:0 while/lstm_cell/Const_7:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/Add_7?
)while/lstm_cell/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2+
)while/lstm_cell/clip_by_value_2/Minimum/y?
'while/lstm_cell/clip_by_value_2/MinimumMinimumwhile/lstm_cell/Add_7:z:02while/lstm_cell/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:??????????2)
'while/lstm_cell/clip_by_value_2/Minimum?
!while/lstm_cell/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!while/lstm_cell/clip_by_value_2/y?
while/lstm_cell/clip_by_value_2Maximum+while/lstm_cell/clip_by_value_2/Minimum:z:0*while/lstm_cell/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:??????????2!
while/lstm_cell/clip_by_value_2?
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/add_5:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/Sigmoid_1?
while/lstm_cell/mul_5Mul#while/lstm_cell/clip_by_value_2:z:0while/lstm_cell/Sigmoid_1:y:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/mul_5?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_5:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:0^while/lstm_cell/ReadVariableOp!^while/lstm_cell/ReadVariableOp_1!^while/lstm_cell/ReadVariableOp_2!^while/lstm_cell/ReadVariableOp_3%^while/lstm_cell/split/ReadVariableOp'^while/lstm_cell/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations^while/lstm_cell/ReadVariableOp!^while/lstm_cell/ReadVariableOp_1!^while/lstm_cell/ReadVariableOp_2!^while/lstm_cell/ReadVariableOp_3%^while/lstm_cell/split/ReadVariableOp'^while/lstm_cell/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0^while/lstm_cell/ReadVariableOp!^while/lstm_cell/ReadVariableOp_1!^while/lstm_cell/ReadVariableOp_2!^while/lstm_cell/ReadVariableOp_3%^while/lstm_cell/split/ReadVariableOp'^while/lstm_cell/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/lstm_cell/ReadVariableOp!^while/lstm_cell/ReadVariableOp_1!^while/lstm_cell/ReadVariableOp_2!^while/lstm_cell/ReadVariableOp_3%^while/lstm_cell/split/ReadVariableOp'^while/lstm_cell/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell/mul_5:z:0^while/lstm_cell/ReadVariableOp!^while/lstm_cell/ReadVariableOp_1!^while/lstm_cell/ReadVariableOp_2!^while/lstm_cell/ReadVariableOp_3%^while/lstm_cell/split/ReadVariableOp'^while/lstm_cell/split_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell/add_5:z:0^while/lstm_cell/ReadVariableOp!^while/lstm_cell/ReadVariableOp_1!^while/lstm_cell/ReadVariableOp_2!^while/lstm_cell/ReadVariableOp_3%^while/lstm_cell/split/ReadVariableOp'^while/lstm_cell/split_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"T
'while_lstm_cell_readvariableop_resource)while_lstm_cell_readvariableop_resource_0"d
/while_lstm_cell_split_1_readvariableop_resource1while_lstm_cell_split_1_readvariableop_resource_0"`
-while_lstm_cell_split_readvariableop_resource/while_lstm_cell_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :??????????:??????????: : :::2@
while/lstm_cell/ReadVariableOpwhile/lstm_cell/ReadVariableOp2D
 while/lstm_cell/ReadVariableOp_1 while/lstm_cell/ReadVariableOp_12D
 while/lstm_cell/ReadVariableOp_2 while/lstm_cell/ReadVariableOp_22D
 while/lstm_cell/ReadVariableOp_3 while/lstm_cell/ReadVariableOp_32L
$while/lstm_cell/split/ReadVariableOp$while/lstm_cell/split/ReadVariableOp2P
&while/lstm_cell/split_1/ReadVariableOp&while/lstm_cell/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_121481
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_121481___redundant_placeholder04
0while_while_cond_121481___redundant_placeholder14
0while_while_cond_121481___redundant_placeholder24
0while_while_cond_121481___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
h
I__inference_Dropout_lay_0_layer_call_and_return_conditional_losses_119983

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
+__inference_sequential_layer_call_fn_120160
lstm_lay_0_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalllstm_lay_0_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????`*)
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_1201432
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????`2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:??????????:::::::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
,
_output_shapes
:??????????
*
_user_specified_nameLSTM_lay_0_input
?
?
while_cond_119798
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_119798___redundant_placeholder04
0while_while_cond_119798___redundant_placeholder14
0while_while_cond_119798___redundant_placeholder24
0while_while_cond_119798___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
??
?
while_body_121750
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_03
/while_lstm_cell_split_readvariableop_resource_05
1while_lstm_cell_split_1_readvariableop_resource_0-
)while_lstm_cell_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor1
-while_lstm_cell_split_readvariableop_resource3
/while_lstm_cell_split_1_readvariableop_resource+
'while_lstm_cell_readvariableop_resource??while/lstm_cell/ReadVariableOp? while/lstm_cell/ReadVariableOp_1? while/lstm_cell/ReadVariableOp_2? while/lstm_cell/ReadVariableOp_3?$while/lstm_cell/split/ReadVariableOp?&while/lstm_cell/split_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemp
while/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell/Const?
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2!
while/lstm_cell/split/split_dim?
$while/lstm_cell/split/ReadVariableOpReadVariableOp/while_lstm_cell_split_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02&
$while/lstm_cell/split/ReadVariableOp?
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0,while/lstm_cell/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
??:
??:
??:
??*
	num_split2
while/lstm_cell/split?
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/MatMul?
while/lstm_cell/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:1*
T0*(
_output_shapes
:??????????2
while/lstm_cell/MatMul_1?
while/lstm_cell/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:2*
T0*(
_output_shapes
:??????????2
while/lstm_cell/MatMul_2?
while/lstm_cell/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:3*
T0*(
_output_shapes
:??????????2
while/lstm_cell/MatMul_3t
while/lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell/Const_1?
!while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!while/lstm_cell/split_1/split_dim?
&while/lstm_cell/split_1/ReadVariableOpReadVariableOp1while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
:?*
dtype02(
&while/lstm_cell/split_1/ReadVariableOp?
while/lstm_cell/split_1Split*while/lstm_cell/split_1/split_dim:output:0.while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2
while/lstm_cell/split_1?
while/lstm_cell/BiasAddBiasAdd while/lstm_cell/MatMul:product:0 while/lstm_cell/split_1:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/BiasAdd?
while/lstm_cell/BiasAdd_1BiasAdd"while/lstm_cell/MatMul_1:product:0 while/lstm_cell/split_1:output:1*
T0*(
_output_shapes
:??????????2
while/lstm_cell/BiasAdd_1?
while/lstm_cell/BiasAdd_2BiasAdd"while/lstm_cell/MatMul_2:product:0 while/lstm_cell/split_1:output:2*
T0*(
_output_shapes
:??????????2
while/lstm_cell/BiasAdd_2?
while/lstm_cell/BiasAdd_3BiasAdd"while/lstm_cell/MatMul_3:product:0 while/lstm_cell/split_1:output:3*
T0*(
_output_shapes
:??????????2
while/lstm_cell/BiasAdd_3?
while/lstm_cell/ReadVariableOpReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02 
while/lstm_cell/ReadVariableOp?
#while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2%
#while/lstm_cell/strided_slice/stack?
%while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2'
%while/lstm_cell/strided_slice/stack_1?
%while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2'
%while/lstm_cell/strided_slice/stack_2?
while/lstm_cell/strided_sliceStridedSlice&while/lstm_cell/ReadVariableOp:value:0,while/lstm_cell/strided_slice/stack:output:0.while/lstm_cell/strided_slice/stack_1:output:0.while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
while/lstm_cell/strided_slice?
while/lstm_cell/MatMul_4MatMulwhile_placeholder_2&while/lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/MatMul_4?
while/lstm_cell/addAddV2 while/lstm_cell/BiasAdd:output:0"while/lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/addw
while/lstm_cell/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/lstm_cell/Const_2w
while/lstm_cell/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/lstm_cell/Const_3?
while/lstm_cell/MulMulwhile/lstm_cell/add:z:0 while/lstm_cell/Const_2:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/Mul?
while/lstm_cell/Add_1Addwhile/lstm_cell/Mul:z:0 while/lstm_cell/Const_3:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/Add_1?
'while/lstm_cell/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2)
'while/lstm_cell/clip_by_value/Minimum/y?
%while/lstm_cell/clip_by_value/MinimumMinimumwhile/lstm_cell/Add_1:z:00while/lstm_cell/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:??????????2'
%while/lstm_cell/clip_by_value/Minimum?
while/lstm_cell/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
while/lstm_cell/clip_by_value/y?
while/lstm_cell/clip_by_valueMaximum)while/lstm_cell/clip_by_value/Minimum:z:0(while/lstm_cell/clip_by_value/y:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/clip_by_value?
 while/lstm_cell/ReadVariableOp_1ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02"
 while/lstm_cell/ReadVariableOp_1?
%while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2'
%while/lstm_cell/strided_slice_1/stack?
'while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2)
'while/lstm_cell/strided_slice_1/stack_1?
'while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_1/stack_2?
while/lstm_cell/strided_slice_1StridedSlice(while/lstm_cell/ReadVariableOp_1:value:0.while/lstm_cell/strided_slice_1/stack:output:00while/lstm_cell/strided_slice_1/stack_1:output:00while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2!
while/lstm_cell/strided_slice_1?
while/lstm_cell/MatMul_5MatMulwhile_placeholder_2(while/lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/MatMul_5?
while/lstm_cell/add_2AddV2"while/lstm_cell/BiasAdd_1:output:0"while/lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/add_2w
while/lstm_cell/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/lstm_cell/Const_4w
while/lstm_cell/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/lstm_cell/Const_5?
while/lstm_cell/Mul_1Mulwhile/lstm_cell/add_2:z:0 while/lstm_cell/Const_4:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/Mul_1?
while/lstm_cell/Add_3Addwhile/lstm_cell/Mul_1:z:0 while/lstm_cell/Const_5:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/Add_3?
)while/lstm_cell/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2+
)while/lstm_cell/clip_by_value_1/Minimum/y?
'while/lstm_cell/clip_by_value_1/MinimumMinimumwhile/lstm_cell/Add_3:z:02while/lstm_cell/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:??????????2)
'while/lstm_cell/clip_by_value_1/Minimum?
!while/lstm_cell/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!while/lstm_cell/clip_by_value_1/y?
while/lstm_cell/clip_by_value_1Maximum+while/lstm_cell/clip_by_value_1/Minimum:z:0*while/lstm_cell/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:??????????2!
while/lstm_cell/clip_by_value_1?
while/lstm_cell/mul_2Mul#while/lstm_cell/clip_by_value_1:z:0while_placeholder_3*
T0*(
_output_shapes
:??????????2
while/lstm_cell/mul_2?
 while/lstm_cell/ReadVariableOp_2ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02"
 while/lstm_cell/ReadVariableOp_2?
%while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2'
%while/lstm_cell/strided_slice_2/stack?
'while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2)
'while/lstm_cell/strided_slice_2/stack_1?
'while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_2/stack_2?
while/lstm_cell/strided_slice_2StridedSlice(while/lstm_cell/ReadVariableOp_2:value:0.while/lstm_cell/strided_slice_2/stack:output:00while/lstm_cell/strided_slice_2/stack_1:output:00while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2!
while/lstm_cell/strided_slice_2?
while/lstm_cell/MatMul_6MatMulwhile_placeholder_2(while/lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/MatMul_6?
while/lstm_cell/add_4AddV2"while/lstm_cell/BiasAdd_2:output:0"while/lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/add_4?
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/add_4:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/Sigmoid?
while/lstm_cell/mul_3Mul!while/lstm_cell/clip_by_value:z:0while/lstm_cell/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/mul_3?
while/lstm_cell/add_5AddV2while/lstm_cell/mul_2:z:0while/lstm_cell/mul_3:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/add_5?
 while/lstm_cell/ReadVariableOp_3ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02"
 while/lstm_cell/ReadVariableOp_3?
%while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2'
%while/lstm_cell/strided_slice_3/stack?
'while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'while/lstm_cell/strided_slice_3/stack_1?
'while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_3/stack_2?
while/lstm_cell/strided_slice_3StridedSlice(while/lstm_cell/ReadVariableOp_3:value:0.while/lstm_cell/strided_slice_3/stack:output:00while/lstm_cell/strided_slice_3/stack_1:output:00while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2!
while/lstm_cell/strided_slice_3?
while/lstm_cell/MatMul_7MatMulwhile_placeholder_2(while/lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/MatMul_7?
while/lstm_cell/add_6AddV2"while/lstm_cell/BiasAdd_3:output:0"while/lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/add_6w
while/lstm_cell/Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/lstm_cell/Const_6w
while/lstm_cell/Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/lstm_cell/Const_7?
while/lstm_cell/Mul_4Mulwhile/lstm_cell/add_6:z:0 while/lstm_cell/Const_6:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/Mul_4?
while/lstm_cell/Add_7Addwhile/lstm_cell/Mul_4:z:0 while/lstm_cell/Const_7:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/Add_7?
)while/lstm_cell/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2+
)while/lstm_cell/clip_by_value_2/Minimum/y?
'while/lstm_cell/clip_by_value_2/MinimumMinimumwhile/lstm_cell/Add_7:z:02while/lstm_cell/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:??????????2)
'while/lstm_cell/clip_by_value_2/Minimum?
!while/lstm_cell/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!while/lstm_cell/clip_by_value_2/y?
while/lstm_cell/clip_by_value_2Maximum+while/lstm_cell/clip_by_value_2/Minimum:z:0*while/lstm_cell/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:??????????2!
while/lstm_cell/clip_by_value_2?
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/add_5:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/Sigmoid_1?
while/lstm_cell/mul_5Mul#while/lstm_cell/clip_by_value_2:z:0while/lstm_cell/Sigmoid_1:y:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/mul_5?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_5:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:0^while/lstm_cell/ReadVariableOp!^while/lstm_cell/ReadVariableOp_1!^while/lstm_cell/ReadVariableOp_2!^while/lstm_cell/ReadVariableOp_3%^while/lstm_cell/split/ReadVariableOp'^while/lstm_cell/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations^while/lstm_cell/ReadVariableOp!^while/lstm_cell/ReadVariableOp_1!^while/lstm_cell/ReadVariableOp_2!^while/lstm_cell/ReadVariableOp_3%^while/lstm_cell/split/ReadVariableOp'^while/lstm_cell/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0^while/lstm_cell/ReadVariableOp!^while/lstm_cell/ReadVariableOp_1!^while/lstm_cell/ReadVariableOp_2!^while/lstm_cell/ReadVariableOp_3%^while/lstm_cell/split/ReadVariableOp'^while/lstm_cell/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/lstm_cell/ReadVariableOp!^while/lstm_cell/ReadVariableOp_1!^while/lstm_cell/ReadVariableOp_2!^while/lstm_cell/ReadVariableOp_3%^while/lstm_cell/split/ReadVariableOp'^while/lstm_cell/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell/mul_5:z:0^while/lstm_cell/ReadVariableOp!^while/lstm_cell/ReadVariableOp_1!^while/lstm_cell/ReadVariableOp_2!^while/lstm_cell/ReadVariableOp_3%^while/lstm_cell/split/ReadVariableOp'^while/lstm_cell/split_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell/add_5:z:0^while/lstm_cell/ReadVariableOp!^while/lstm_cell/ReadVariableOp_1!^while/lstm_cell/ReadVariableOp_2!^while/lstm_cell/ReadVariableOp_3%^while/lstm_cell/split/ReadVariableOp'^while/lstm_cell/split_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"T
'while_lstm_cell_readvariableop_resource)while_lstm_cell_readvariableop_resource_0"d
/while_lstm_cell_split_1_readvariableop_resource1while_lstm_cell_split_1_readvariableop_resource_0"`
-while_lstm_cell_split_readvariableop_resource/while_lstm_cell_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :??????????:??????????: : :::2@
while/lstm_cell/ReadVariableOpwhile/lstm_cell/ReadVariableOp2D
 while/lstm_cell/ReadVariableOp_1 while/lstm_cell/ReadVariableOp_12D
 while/lstm_cell/ReadVariableOp_2 while/lstm_cell/ReadVariableOp_22D
 while/lstm_cell/ReadVariableOp_3 while/lstm_cell/ReadVariableOp_32L
$while/lstm_cell/split/ReadVariableOp$while/lstm_cell/split/ReadVariableOp2P
&while/lstm_cell/split_1/ReadVariableOp&while/lstm_cell/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_119193
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_119193___redundant_placeholder04
0while_while_cond_119193___redundant_placeholder14
0while_while_cond_119193___redundant_placeholder24
0while_while_cond_119193___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
??
?
F__inference_LSTM_lay_0_layer_call_and_return_conditional_losses_121066

inputs+
'lstm_cell_split_readvariableop_resource-
)lstm_cell_split_1_readvariableop_resource%
!lstm_cell_readvariableop_resource
identity??lstm_cell/ReadVariableOp?lstm_cell/ReadVariableOp_1?lstm_cell/ReadVariableOp_2?lstm_cell/ReadVariableOp_3?lstm_cell/split/ReadVariableOp? lstm_cell/split_1/ReadVariableOp?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:??????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_2d
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dim?
lstm_cell/split/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
lstm_cell/split/ReadVariableOp?
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
??:
??:
??:
??*
	num_split2
lstm_cell/split?
lstm_cell/MatMulMatMulstrided_slice_2:output:0lstm_cell/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul?
lstm_cell/MatMul_1MatMulstrided_slice_2:output:0lstm_cell/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_1?
lstm_cell/MatMul_2MatMulstrided_slice_2:output:0lstm_cell/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_2?
lstm_cell/MatMul_3MatMulstrided_slice_2:output:0lstm_cell/split:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_3h
lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Const_1|
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell/split_1/split_dim?
 lstm_cell/split_1/ReadVariableOpReadVariableOp)lstm_cell_split_1_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 lstm_cell/split_1/ReadVariableOp?
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2
lstm_cell/split_1?
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd?
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_1?
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_2?
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_3?
lstm_cell/ReadVariableOpReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp?
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
lstm_cell/strided_slice/stack?
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2!
lstm_cell/strided_slice/stack_1?
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
lstm_cell/strided_slice/stack_2?
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice?
lstm_cell/MatMul_4MatMulzeros:output:0 lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_4?
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/addk
lstm_cell/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
lstm_cell/Const_2k
lstm_cell/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm_cell/Const_3?
lstm_cell/MulMullstm_cell/add:z:0lstm_cell/Const_2:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Mul?
lstm_cell/Add_1Addlstm_cell/Mul:z:0lstm_cell/Const_3:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Add_1?
!lstm_cell/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2#
!lstm_cell/clip_by_value/Minimum/y?
lstm_cell/clip_by_value/MinimumMinimumlstm_cell/Add_1:z:0*lstm_cell/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:??????????2!
lstm_cell/clip_by_value/Minimum{
lstm_cell/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_cell/clip_by_value/y?
lstm_cell/clip_by_valueMaximum#lstm_cell/clip_by_value/Minimum:z:0"lstm_cell/clip_by_value/y:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/clip_by_value?
lstm_cell/ReadVariableOp_1ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_1?
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2!
lstm_cell/strided_slice_1/stack?
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_1/stack_1?
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_1/stack_2?
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_1?
lstm_cell/MatMul_5MatMulzeros:output:0"lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_5?
lstm_cell/add_2AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_2k
lstm_cell/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
lstm_cell/Const_4k
lstm_cell/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm_cell/Const_5?
lstm_cell/Mul_1Mullstm_cell/add_2:z:0lstm_cell/Const_4:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Mul_1?
lstm_cell/Add_3Addlstm_cell/Mul_1:z:0lstm_cell/Const_5:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Add_3?
#lstm_cell/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2%
#lstm_cell/clip_by_value_1/Minimum/y?
!lstm_cell/clip_by_value_1/MinimumMinimumlstm_cell/Add_3:z:0,lstm_cell/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:??????????2#
!lstm_cell/clip_by_value_1/Minimum
lstm_cell/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_cell/clip_by_value_1/y?
lstm_cell/clip_by_value_1Maximum%lstm_cell/clip_by_value_1/Minimum:z:0$lstm_cell/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/clip_by_value_1?
lstm_cell/mul_2Mullstm_cell/clip_by_value_1:z:0zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_2?
lstm_cell/ReadVariableOp_2ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_2?
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_2/stack?
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2#
!lstm_cell/strided_slice_2/stack_1?
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_2/stack_2?
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_2?
lstm_cell/MatMul_6MatMulzeros:output:0"lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_6?
lstm_cell/add_4AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_4y
lstm_cell/SigmoidSigmoidlstm_cell/add_4:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid?
lstm_cell/mul_3Mullstm_cell/clip_by_value:z:0lstm_cell/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_3?
lstm_cell/add_5AddV2lstm_cell/mul_2:z:0lstm_cell/mul_3:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_5?
lstm_cell/ReadVariableOp_3ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_3?
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2!
lstm_cell/strided_slice_3/stack?
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_3/stack_1?
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_3/stack_2?
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_3?
lstm_cell/MatMul_7MatMulzeros:output:0"lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_7?
lstm_cell/add_6AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_6k
lstm_cell/Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
lstm_cell/Const_6k
lstm_cell/Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm_cell/Const_7?
lstm_cell/Mul_4Mullstm_cell/add_6:z:0lstm_cell/Const_6:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Mul_4?
lstm_cell/Add_7Addlstm_cell/Mul_4:z:0lstm_cell/Const_7:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Add_7?
#lstm_cell/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2%
#lstm_cell/clip_by_value_2/Minimum/y?
!lstm_cell/clip_by_value_2/MinimumMinimumlstm_cell/Add_7:z:0,lstm_cell/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:??????????2#
!lstm_cell/clip_by_value_2/Minimum
lstm_cell/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_cell/clip_by_value_2/y?
lstm_cell/clip_by_value_2Maximum%lstm_cell/clip_by_value_2/Minimum:z:0$lstm_cell/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/clip_by_value_2}
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_5:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid_1?
lstm_cell/mul_5Mullstm_cell/clip_by_value_2:z:0lstm_cell/Sigmoid_1:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_5?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0'lstm_cell_split_readvariableop_resource)lstm_cell_split_1_readvariableop_resource!lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_120924*
condR
while_cond_120923*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2
transpose_1?
IdentityIdentitystrided_slice_3:output:0^lstm_cell/ReadVariableOp^lstm_cell/ReadVariableOp_1^lstm_cell/ReadVariableOp_2^lstm_cell/ReadVariableOp_3^lstm_cell/split/ReadVariableOp!^lstm_cell/split_1/ReadVariableOp^while*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????:::24
lstm_cell/ReadVariableOplstm_cell/ReadVariableOp28
lstm_cell/ReadVariableOp_1lstm_cell/ReadVariableOp_128
lstm_cell/ReadVariableOp_2lstm_cell/ReadVariableOp_228
lstm_cell/ReadVariableOp_3lstm_cell/ReadVariableOp_32@
lstm_cell/split/ReadVariableOplstm_cell/split/ReadVariableOp2D
 lstm_cell/split_1/ReadVariableOp lstm_cell/split_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
+__inference_LSTM_lay_0_layer_call_fn_121345

inputs
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_LSTM_lay_0_layer_call_and_return_conditional_losses_1196732
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????:::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
ۡ
?
F__inference_LSTM_lay_0_layer_call_and_return_conditional_losses_121892
inputs_0+
'lstm_cell_split_readvariableop_resource-
)lstm_cell_split_1_readvariableop_resource%
!lstm_cell_readvariableop_resource
identity??lstm_cell/ReadVariableOp?lstm_cell/ReadVariableOp_1?lstm_cell/ReadVariableOp_2?lstm_cell/ReadVariableOp_3?lstm_cell/split/ReadVariableOp? lstm_cell/split_1/ReadVariableOp?whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_2d
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dim?
lstm_cell/split/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
lstm_cell/split/ReadVariableOp?
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
??:
??:
??:
??*
	num_split2
lstm_cell/split?
lstm_cell/MatMulMatMulstrided_slice_2:output:0lstm_cell/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul?
lstm_cell/MatMul_1MatMulstrided_slice_2:output:0lstm_cell/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_1?
lstm_cell/MatMul_2MatMulstrided_slice_2:output:0lstm_cell/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_2?
lstm_cell/MatMul_3MatMulstrided_slice_2:output:0lstm_cell/split:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_3h
lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Const_1|
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell/split_1/split_dim?
 lstm_cell/split_1/ReadVariableOpReadVariableOp)lstm_cell_split_1_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 lstm_cell/split_1/ReadVariableOp?
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2
lstm_cell/split_1?
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd?
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_1?
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_2?
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell/BiasAdd_3?
lstm_cell/ReadVariableOpReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp?
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
lstm_cell/strided_slice/stack?
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2!
lstm_cell/strided_slice/stack_1?
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
lstm_cell/strided_slice/stack_2?
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice?
lstm_cell/MatMul_4MatMulzeros:output:0 lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_4?
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/addk
lstm_cell/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
lstm_cell/Const_2k
lstm_cell/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm_cell/Const_3?
lstm_cell/MulMullstm_cell/add:z:0lstm_cell/Const_2:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Mul?
lstm_cell/Add_1Addlstm_cell/Mul:z:0lstm_cell/Const_3:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Add_1?
!lstm_cell/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2#
!lstm_cell/clip_by_value/Minimum/y?
lstm_cell/clip_by_value/MinimumMinimumlstm_cell/Add_1:z:0*lstm_cell/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:??????????2!
lstm_cell/clip_by_value/Minimum{
lstm_cell/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_cell/clip_by_value/y?
lstm_cell/clip_by_valueMaximum#lstm_cell/clip_by_value/Minimum:z:0"lstm_cell/clip_by_value/y:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/clip_by_value?
lstm_cell/ReadVariableOp_1ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_1?
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2!
lstm_cell/strided_slice_1/stack?
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!lstm_cell/strided_slice_1/stack_1?
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_1/stack_2?
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_1?
lstm_cell/MatMul_5MatMulzeros:output:0"lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_5?
lstm_cell/add_2AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_2k
lstm_cell/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
lstm_cell/Const_4k
lstm_cell/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm_cell/Const_5?
lstm_cell/Mul_1Mullstm_cell/add_2:z:0lstm_cell/Const_4:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Mul_1?
lstm_cell/Add_3Addlstm_cell/Mul_1:z:0lstm_cell/Const_5:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Add_3?
#lstm_cell/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2%
#lstm_cell/clip_by_value_1/Minimum/y?
!lstm_cell/clip_by_value_1/MinimumMinimumlstm_cell/Add_3:z:0,lstm_cell/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:??????????2#
!lstm_cell/clip_by_value_1/Minimum
lstm_cell/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_cell/clip_by_value_1/y?
lstm_cell/clip_by_value_1Maximum%lstm_cell/clip_by_value_1/Minimum:z:0$lstm_cell/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/clip_by_value_1?
lstm_cell/mul_2Mullstm_cell/clip_by_value_1:z:0zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_2?
lstm_cell/ReadVariableOp_2ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_2?
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
lstm_cell/strided_slice_2/stack?
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2#
!lstm_cell/strided_slice_2/stack_1?
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_2/stack_2?
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_2?
lstm_cell/MatMul_6MatMulzeros:output:0"lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_6?
lstm_cell/add_4AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_4y
lstm_cell/SigmoidSigmoidlstm_cell/add_4:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid?
lstm_cell/mul_3Mullstm_cell/clip_by_value:z:0lstm_cell/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_3?
lstm_cell/add_5AddV2lstm_cell/mul_2:z:0lstm_cell/mul_3:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_5?
lstm_cell/ReadVariableOp_3ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
??*
dtype02
lstm_cell/ReadVariableOp_3?
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2!
lstm_cell/strided_slice_3/stack?
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_3/stack_1?
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_3/stack_2?
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
lstm_cell/strided_slice_3?
lstm_cell/MatMul_7MatMulzeros:output:0"lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/MatMul_7?
lstm_cell/add_6AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell/add_6k
lstm_cell/Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
lstm_cell/Const_6k
lstm_cell/Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm_cell/Const_7?
lstm_cell/Mul_4Mullstm_cell/add_6:z:0lstm_cell/Const_6:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Mul_4?
lstm_cell/Add_7Addlstm_cell/Mul_4:z:0lstm_cell/Const_7:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Add_7?
#lstm_cell/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2%
#lstm_cell/clip_by_value_2/Minimum/y?
!lstm_cell/clip_by_value_2/MinimumMinimumlstm_cell/Add_7:z:0,lstm_cell/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:??????????2#
!lstm_cell/clip_by_value_2/Minimum
lstm_cell/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_cell/clip_by_value_2/y?
lstm_cell/clip_by_value_2Maximum%lstm_cell/clip_by_value_2/Minimum:z:0$lstm_cell/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell/clip_by_value_2}
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_5:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell/Sigmoid_1?
lstm_cell/mul_5Mullstm_cell/clip_by_value_2:z:0lstm_cell/Sigmoid_1:y:0*
T0*(
_output_shapes
:??????????2
lstm_cell/mul_5?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0'lstm_cell_split_readvariableop_resource)lstm_cell_split_1_readvariableop_resource!lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_121750*
condR
while_cond_121749*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
transpose_1?
IdentityIdentitystrided_slice_3:output:0^lstm_cell/ReadVariableOp^lstm_cell/ReadVariableOp_1^lstm_cell/ReadVariableOp_2^lstm_cell/ReadVariableOp_3^lstm_cell/split/ReadVariableOp!^lstm_cell/split_1/ReadVariableOp^while*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:???????????????????:::24
lstm_cell/ReadVariableOplstm_cell/ReadVariableOp28
lstm_cell/ReadVariableOp_1lstm_cell/ReadVariableOp_128
lstm_cell/ReadVariableOp_2lstm_cell/ReadVariableOp_228
lstm_cell/ReadVariableOp_3lstm_cell/ReadVariableOp_32@
lstm_cell/split/ReadVariableOplstm_cell/split/ReadVariableOp2D
 lstm_cell/split_1/ReadVariableOp lstm_cell/split_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:???????????????????
"
_user_specified_name
inputs/0
??
?
LSTM_lay_0_while_body_1203152
.lstm_lay_0_while_lstm_lay_0_while_loop_counter8
4lstm_lay_0_while_lstm_lay_0_while_maximum_iterations 
lstm_lay_0_while_placeholder"
lstm_lay_0_while_placeholder_1"
lstm_lay_0_while_placeholder_2"
lstm_lay_0_while_placeholder_31
-lstm_lay_0_while_lstm_lay_0_strided_slice_1_0m
ilstm_lay_0_while_tensorarrayv2read_tensorlistgetitem_lstm_lay_0_tensorarrayunstack_tensorlistfromtensor_0>
:lstm_lay_0_while_lstm_cell_split_readvariableop_resource_0@
<lstm_lay_0_while_lstm_cell_split_1_readvariableop_resource_08
4lstm_lay_0_while_lstm_cell_readvariableop_resource_0
lstm_lay_0_while_identity
lstm_lay_0_while_identity_1
lstm_lay_0_while_identity_2
lstm_lay_0_while_identity_3
lstm_lay_0_while_identity_4
lstm_lay_0_while_identity_5/
+lstm_lay_0_while_lstm_lay_0_strided_slice_1k
glstm_lay_0_while_tensorarrayv2read_tensorlistgetitem_lstm_lay_0_tensorarrayunstack_tensorlistfromtensor<
8lstm_lay_0_while_lstm_cell_split_readvariableop_resource>
:lstm_lay_0_while_lstm_cell_split_1_readvariableop_resource6
2lstm_lay_0_while_lstm_cell_readvariableop_resource??)LSTM_lay_0/while/lstm_cell/ReadVariableOp?+LSTM_lay_0/while/lstm_cell/ReadVariableOp_1?+LSTM_lay_0/while/lstm_cell/ReadVariableOp_2?+LSTM_lay_0/while/lstm_cell/ReadVariableOp_3?/LSTM_lay_0/while/lstm_cell/split/ReadVariableOp?1LSTM_lay_0/while/lstm_cell/split_1/ReadVariableOp?
BLSTM_lay_0/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2D
BLSTM_lay_0/while/TensorArrayV2Read/TensorListGetItem/element_shape?
4LSTM_lay_0/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemilstm_lay_0_while_tensorarrayv2read_tensorlistgetitem_lstm_lay_0_tensorarrayunstack_tensorlistfromtensor_0lstm_lay_0_while_placeholderKLSTM_lay_0/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype026
4LSTM_lay_0/while/TensorArrayV2Read/TensorListGetItem?
 LSTM_lay_0/while/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2"
 LSTM_lay_0/while/lstm_cell/Const?
*LSTM_lay_0/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*LSTM_lay_0/while/lstm_cell/split/split_dim?
/LSTM_lay_0/while/lstm_cell/split/ReadVariableOpReadVariableOp:lstm_lay_0_while_lstm_cell_split_readvariableop_resource_0* 
_output_shapes
:
??*
dtype021
/LSTM_lay_0/while/lstm_cell/split/ReadVariableOp?
 LSTM_lay_0/while/lstm_cell/splitSplit3LSTM_lay_0/while/lstm_cell/split/split_dim:output:07LSTM_lay_0/while/lstm_cell/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
??:
??:
??:
??*
	num_split2"
 LSTM_lay_0/while/lstm_cell/split?
!LSTM_lay_0/while/lstm_cell/MatMulMatMul;LSTM_lay_0/while/TensorArrayV2Read/TensorListGetItem:item:0)LSTM_lay_0/while/lstm_cell/split:output:0*
T0*(
_output_shapes
:??????????2#
!LSTM_lay_0/while/lstm_cell/MatMul?
#LSTM_lay_0/while/lstm_cell/MatMul_1MatMul;LSTM_lay_0/while/TensorArrayV2Read/TensorListGetItem:item:0)LSTM_lay_0/while/lstm_cell/split:output:1*
T0*(
_output_shapes
:??????????2%
#LSTM_lay_0/while/lstm_cell/MatMul_1?
#LSTM_lay_0/while/lstm_cell/MatMul_2MatMul;LSTM_lay_0/while/TensorArrayV2Read/TensorListGetItem:item:0)LSTM_lay_0/while/lstm_cell/split:output:2*
T0*(
_output_shapes
:??????????2%
#LSTM_lay_0/while/lstm_cell/MatMul_2?
#LSTM_lay_0/while/lstm_cell/MatMul_3MatMul;LSTM_lay_0/while/TensorArrayV2Read/TensorListGetItem:item:0)LSTM_lay_0/while/lstm_cell/split:output:3*
T0*(
_output_shapes
:??????????2%
#LSTM_lay_0/while/lstm_cell/MatMul_3?
"LSTM_lay_0/while/lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2$
"LSTM_lay_0/while/lstm_cell/Const_1?
,LSTM_lay_0/while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2.
,LSTM_lay_0/while/lstm_cell/split_1/split_dim?
1LSTM_lay_0/while/lstm_cell/split_1/ReadVariableOpReadVariableOp<lstm_lay_0_while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
:?*
dtype023
1LSTM_lay_0/while/lstm_cell/split_1/ReadVariableOp?
"LSTM_lay_0/while/lstm_cell/split_1Split5LSTM_lay_0/while/lstm_cell/split_1/split_dim:output:09LSTM_lay_0/while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2$
"LSTM_lay_0/while/lstm_cell/split_1?
"LSTM_lay_0/while/lstm_cell/BiasAddBiasAdd+LSTM_lay_0/while/lstm_cell/MatMul:product:0+LSTM_lay_0/while/lstm_cell/split_1:output:0*
T0*(
_output_shapes
:??????????2$
"LSTM_lay_0/while/lstm_cell/BiasAdd?
$LSTM_lay_0/while/lstm_cell/BiasAdd_1BiasAdd-LSTM_lay_0/while/lstm_cell/MatMul_1:product:0+LSTM_lay_0/while/lstm_cell/split_1:output:1*
T0*(
_output_shapes
:??????????2&
$LSTM_lay_0/while/lstm_cell/BiasAdd_1?
$LSTM_lay_0/while/lstm_cell/BiasAdd_2BiasAdd-LSTM_lay_0/while/lstm_cell/MatMul_2:product:0+LSTM_lay_0/while/lstm_cell/split_1:output:2*
T0*(
_output_shapes
:??????????2&
$LSTM_lay_0/while/lstm_cell/BiasAdd_2?
$LSTM_lay_0/while/lstm_cell/BiasAdd_3BiasAdd-LSTM_lay_0/while/lstm_cell/MatMul_3:product:0+LSTM_lay_0/while/lstm_cell/split_1:output:3*
T0*(
_output_shapes
:??????????2&
$LSTM_lay_0/while/lstm_cell/BiasAdd_3?
)LSTM_lay_0/while/lstm_cell/ReadVariableOpReadVariableOp4lstm_lay_0_while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02+
)LSTM_lay_0/while/lstm_cell/ReadVariableOp?
.LSTM_lay_0/while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        20
.LSTM_lay_0/while/lstm_cell/strided_slice/stack?
0LSTM_lay_0/while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   22
0LSTM_lay_0/while/lstm_cell/strided_slice/stack_1?
0LSTM_lay_0/while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0LSTM_lay_0/while/lstm_cell/strided_slice/stack_2?
(LSTM_lay_0/while/lstm_cell/strided_sliceStridedSlice1LSTM_lay_0/while/lstm_cell/ReadVariableOp:value:07LSTM_lay_0/while/lstm_cell/strided_slice/stack:output:09LSTM_lay_0/while/lstm_cell/strided_slice/stack_1:output:09LSTM_lay_0/while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2*
(LSTM_lay_0/while/lstm_cell/strided_slice?
#LSTM_lay_0/while/lstm_cell/MatMul_4MatMullstm_lay_0_while_placeholder_21LSTM_lay_0/while/lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:??????????2%
#LSTM_lay_0/while/lstm_cell/MatMul_4?
LSTM_lay_0/while/lstm_cell/addAddV2+LSTM_lay_0/while/lstm_cell/BiasAdd:output:0-LSTM_lay_0/while/lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2 
LSTM_lay_0/while/lstm_cell/add?
"LSTM_lay_0/while/lstm_cell/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2$
"LSTM_lay_0/while/lstm_cell/Const_2?
"LSTM_lay_0/while/lstm_cell/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2$
"LSTM_lay_0/while/lstm_cell/Const_3?
LSTM_lay_0/while/lstm_cell/MulMul"LSTM_lay_0/while/lstm_cell/add:z:0+LSTM_lay_0/while/lstm_cell/Const_2:output:0*
T0*(
_output_shapes
:??????????2 
LSTM_lay_0/while/lstm_cell/Mul?
 LSTM_lay_0/while/lstm_cell/Add_1Add"LSTM_lay_0/while/lstm_cell/Mul:z:0+LSTM_lay_0/while/lstm_cell/Const_3:output:0*
T0*(
_output_shapes
:??????????2"
 LSTM_lay_0/while/lstm_cell/Add_1?
2LSTM_lay_0/while/lstm_cell/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??24
2LSTM_lay_0/while/lstm_cell/clip_by_value/Minimum/y?
0LSTM_lay_0/while/lstm_cell/clip_by_value/MinimumMinimum$LSTM_lay_0/while/lstm_cell/Add_1:z:0;LSTM_lay_0/while/lstm_cell/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:??????????22
0LSTM_lay_0/while/lstm_cell/clip_by_value/Minimum?
*LSTM_lay_0/while/lstm_cell/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*LSTM_lay_0/while/lstm_cell/clip_by_value/y?
(LSTM_lay_0/while/lstm_cell/clip_by_valueMaximum4LSTM_lay_0/while/lstm_cell/clip_by_value/Minimum:z:03LSTM_lay_0/while/lstm_cell/clip_by_value/y:output:0*
T0*(
_output_shapes
:??????????2*
(LSTM_lay_0/while/lstm_cell/clip_by_value?
+LSTM_lay_0/while/lstm_cell/ReadVariableOp_1ReadVariableOp4lstm_lay_0_while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02-
+LSTM_lay_0/while/lstm_cell/ReadVariableOp_1?
0LSTM_lay_0/while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   22
0LSTM_lay_0/while/lstm_cell/strided_slice_1/stack?
2LSTM_lay_0/while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       24
2LSTM_lay_0/while/lstm_cell/strided_slice_1/stack_1?
2LSTM_lay_0/while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      24
2LSTM_lay_0/while/lstm_cell/strided_slice_1/stack_2?
*LSTM_lay_0/while/lstm_cell/strided_slice_1StridedSlice3LSTM_lay_0/while/lstm_cell/ReadVariableOp_1:value:09LSTM_lay_0/while/lstm_cell/strided_slice_1/stack:output:0;LSTM_lay_0/while/lstm_cell/strided_slice_1/stack_1:output:0;LSTM_lay_0/while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2,
*LSTM_lay_0/while/lstm_cell/strided_slice_1?
#LSTM_lay_0/while/lstm_cell/MatMul_5MatMullstm_lay_0_while_placeholder_23LSTM_lay_0/while/lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2%
#LSTM_lay_0/while/lstm_cell/MatMul_5?
 LSTM_lay_0/while/lstm_cell/add_2AddV2-LSTM_lay_0/while/lstm_cell/BiasAdd_1:output:0-LSTM_lay_0/while/lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2"
 LSTM_lay_0/while/lstm_cell/add_2?
"LSTM_lay_0/while/lstm_cell/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2$
"LSTM_lay_0/while/lstm_cell/Const_4?
"LSTM_lay_0/while/lstm_cell/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2$
"LSTM_lay_0/while/lstm_cell/Const_5?
 LSTM_lay_0/while/lstm_cell/Mul_1Mul$LSTM_lay_0/while/lstm_cell/add_2:z:0+LSTM_lay_0/while/lstm_cell/Const_4:output:0*
T0*(
_output_shapes
:??????????2"
 LSTM_lay_0/while/lstm_cell/Mul_1?
 LSTM_lay_0/while/lstm_cell/Add_3Add$LSTM_lay_0/while/lstm_cell/Mul_1:z:0+LSTM_lay_0/while/lstm_cell/Const_5:output:0*
T0*(
_output_shapes
:??????????2"
 LSTM_lay_0/while/lstm_cell/Add_3?
4LSTM_lay_0/while/lstm_cell/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??26
4LSTM_lay_0/while/lstm_cell/clip_by_value_1/Minimum/y?
2LSTM_lay_0/while/lstm_cell/clip_by_value_1/MinimumMinimum$LSTM_lay_0/while/lstm_cell/Add_3:z:0=LSTM_lay_0/while/lstm_cell/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:??????????24
2LSTM_lay_0/while/lstm_cell/clip_by_value_1/Minimum?
,LSTM_lay_0/while/lstm_cell/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2.
,LSTM_lay_0/while/lstm_cell/clip_by_value_1/y?
*LSTM_lay_0/while/lstm_cell/clip_by_value_1Maximum6LSTM_lay_0/while/lstm_cell/clip_by_value_1/Minimum:z:05LSTM_lay_0/while/lstm_cell/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:??????????2,
*LSTM_lay_0/while/lstm_cell/clip_by_value_1?
 LSTM_lay_0/while/lstm_cell/mul_2Mul.LSTM_lay_0/while/lstm_cell/clip_by_value_1:z:0lstm_lay_0_while_placeholder_3*
T0*(
_output_shapes
:??????????2"
 LSTM_lay_0/while/lstm_cell/mul_2?
+LSTM_lay_0/while/lstm_cell/ReadVariableOp_2ReadVariableOp4lstm_lay_0_while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02-
+LSTM_lay_0/while/lstm_cell/ReadVariableOp_2?
0LSTM_lay_0/while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       22
0LSTM_lay_0/while/lstm_cell/strided_slice_2/stack?
2LSTM_lay_0/while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  24
2LSTM_lay_0/while/lstm_cell/strided_slice_2/stack_1?
2LSTM_lay_0/while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      24
2LSTM_lay_0/while/lstm_cell/strided_slice_2/stack_2?
*LSTM_lay_0/while/lstm_cell/strided_slice_2StridedSlice3LSTM_lay_0/while/lstm_cell/ReadVariableOp_2:value:09LSTM_lay_0/while/lstm_cell/strided_slice_2/stack:output:0;LSTM_lay_0/while/lstm_cell/strided_slice_2/stack_1:output:0;LSTM_lay_0/while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2,
*LSTM_lay_0/while/lstm_cell/strided_slice_2?
#LSTM_lay_0/while/lstm_cell/MatMul_6MatMullstm_lay_0_while_placeholder_23LSTM_lay_0/while/lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2%
#LSTM_lay_0/while/lstm_cell/MatMul_6?
 LSTM_lay_0/while/lstm_cell/add_4AddV2-LSTM_lay_0/while/lstm_cell/BiasAdd_2:output:0-LSTM_lay_0/while/lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2"
 LSTM_lay_0/while/lstm_cell/add_4?
"LSTM_lay_0/while/lstm_cell/SigmoidSigmoid$LSTM_lay_0/while/lstm_cell/add_4:z:0*
T0*(
_output_shapes
:??????????2$
"LSTM_lay_0/while/lstm_cell/Sigmoid?
 LSTM_lay_0/while/lstm_cell/mul_3Mul,LSTM_lay_0/while/lstm_cell/clip_by_value:z:0&LSTM_lay_0/while/lstm_cell/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2"
 LSTM_lay_0/while/lstm_cell/mul_3?
 LSTM_lay_0/while/lstm_cell/add_5AddV2$LSTM_lay_0/while/lstm_cell/mul_2:z:0$LSTM_lay_0/while/lstm_cell/mul_3:z:0*
T0*(
_output_shapes
:??????????2"
 LSTM_lay_0/while/lstm_cell/add_5?
+LSTM_lay_0/while/lstm_cell/ReadVariableOp_3ReadVariableOp4lstm_lay_0_while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02-
+LSTM_lay_0/while/lstm_cell/ReadVariableOp_3?
0LSTM_lay_0/while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  22
0LSTM_lay_0/while/lstm_cell/strided_slice_3/stack?
2LSTM_lay_0/while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        24
2LSTM_lay_0/while/lstm_cell/strided_slice_3/stack_1?
2LSTM_lay_0/while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      24
2LSTM_lay_0/while/lstm_cell/strided_slice_3/stack_2?
*LSTM_lay_0/while/lstm_cell/strided_slice_3StridedSlice3LSTM_lay_0/while/lstm_cell/ReadVariableOp_3:value:09LSTM_lay_0/while/lstm_cell/strided_slice_3/stack:output:0;LSTM_lay_0/while/lstm_cell/strided_slice_3/stack_1:output:0;LSTM_lay_0/while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2,
*LSTM_lay_0/while/lstm_cell/strided_slice_3?
#LSTM_lay_0/while/lstm_cell/MatMul_7MatMullstm_lay_0_while_placeholder_23LSTM_lay_0/while/lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2%
#LSTM_lay_0/while/lstm_cell/MatMul_7?
 LSTM_lay_0/while/lstm_cell/add_6AddV2-LSTM_lay_0/while/lstm_cell/BiasAdd_3:output:0-LSTM_lay_0/while/lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2"
 LSTM_lay_0/while/lstm_cell/add_6?
"LSTM_lay_0/while/lstm_cell/Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2$
"LSTM_lay_0/while/lstm_cell/Const_6?
"LSTM_lay_0/while/lstm_cell/Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2$
"LSTM_lay_0/while/lstm_cell/Const_7?
 LSTM_lay_0/while/lstm_cell/Mul_4Mul$LSTM_lay_0/while/lstm_cell/add_6:z:0+LSTM_lay_0/while/lstm_cell/Const_6:output:0*
T0*(
_output_shapes
:??????????2"
 LSTM_lay_0/while/lstm_cell/Mul_4?
 LSTM_lay_0/while/lstm_cell/Add_7Add$LSTM_lay_0/while/lstm_cell/Mul_4:z:0+LSTM_lay_0/while/lstm_cell/Const_7:output:0*
T0*(
_output_shapes
:??????????2"
 LSTM_lay_0/while/lstm_cell/Add_7?
4LSTM_lay_0/while/lstm_cell/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??26
4LSTM_lay_0/while/lstm_cell/clip_by_value_2/Minimum/y?
2LSTM_lay_0/while/lstm_cell/clip_by_value_2/MinimumMinimum$LSTM_lay_0/while/lstm_cell/Add_7:z:0=LSTM_lay_0/while/lstm_cell/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:??????????24
2LSTM_lay_0/while/lstm_cell/clip_by_value_2/Minimum?
,LSTM_lay_0/while/lstm_cell/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2.
,LSTM_lay_0/while/lstm_cell/clip_by_value_2/y?
*LSTM_lay_0/while/lstm_cell/clip_by_value_2Maximum6LSTM_lay_0/while/lstm_cell/clip_by_value_2/Minimum:z:05LSTM_lay_0/while/lstm_cell/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:??????????2,
*LSTM_lay_0/while/lstm_cell/clip_by_value_2?
$LSTM_lay_0/while/lstm_cell/Sigmoid_1Sigmoid$LSTM_lay_0/while/lstm_cell/add_5:z:0*
T0*(
_output_shapes
:??????????2&
$LSTM_lay_0/while/lstm_cell/Sigmoid_1?
 LSTM_lay_0/while/lstm_cell/mul_5Mul.LSTM_lay_0/while/lstm_cell/clip_by_value_2:z:0(LSTM_lay_0/while/lstm_cell/Sigmoid_1:y:0*
T0*(
_output_shapes
:??????????2"
 LSTM_lay_0/while/lstm_cell/mul_5?
5LSTM_lay_0/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_lay_0_while_placeholder_1lstm_lay_0_while_placeholder$LSTM_lay_0/while/lstm_cell/mul_5:z:0*
_output_shapes
: *
element_dtype027
5LSTM_lay_0/while/TensorArrayV2Write/TensorListSetItemr
LSTM_lay_0/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
LSTM_lay_0/while/add/y?
LSTM_lay_0/while/addAddV2lstm_lay_0_while_placeholderLSTM_lay_0/while/add/y:output:0*
T0*
_output_shapes
: 2
LSTM_lay_0/while/addv
LSTM_lay_0/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
LSTM_lay_0/while/add_1/y?
LSTM_lay_0/while/add_1AddV2.lstm_lay_0_while_lstm_lay_0_while_loop_counter!LSTM_lay_0/while/add_1/y:output:0*
T0*
_output_shapes
: 2
LSTM_lay_0/while/add_1?
LSTM_lay_0/while/IdentityIdentityLSTM_lay_0/while/add_1:z:0*^LSTM_lay_0/while/lstm_cell/ReadVariableOp,^LSTM_lay_0/while/lstm_cell/ReadVariableOp_1,^LSTM_lay_0/while/lstm_cell/ReadVariableOp_2,^LSTM_lay_0/while/lstm_cell/ReadVariableOp_30^LSTM_lay_0/while/lstm_cell/split/ReadVariableOp2^LSTM_lay_0/while/lstm_cell/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
LSTM_lay_0/while/Identity?
LSTM_lay_0/while/Identity_1Identity4lstm_lay_0_while_lstm_lay_0_while_maximum_iterations*^LSTM_lay_0/while/lstm_cell/ReadVariableOp,^LSTM_lay_0/while/lstm_cell/ReadVariableOp_1,^LSTM_lay_0/while/lstm_cell/ReadVariableOp_2,^LSTM_lay_0/while/lstm_cell/ReadVariableOp_30^LSTM_lay_0/while/lstm_cell/split/ReadVariableOp2^LSTM_lay_0/while/lstm_cell/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
LSTM_lay_0/while/Identity_1?
LSTM_lay_0/while/Identity_2IdentityLSTM_lay_0/while/add:z:0*^LSTM_lay_0/while/lstm_cell/ReadVariableOp,^LSTM_lay_0/while/lstm_cell/ReadVariableOp_1,^LSTM_lay_0/while/lstm_cell/ReadVariableOp_2,^LSTM_lay_0/while/lstm_cell/ReadVariableOp_30^LSTM_lay_0/while/lstm_cell/split/ReadVariableOp2^LSTM_lay_0/while/lstm_cell/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
LSTM_lay_0/while/Identity_2?
LSTM_lay_0/while/Identity_3IdentityELSTM_lay_0/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^LSTM_lay_0/while/lstm_cell/ReadVariableOp,^LSTM_lay_0/while/lstm_cell/ReadVariableOp_1,^LSTM_lay_0/while/lstm_cell/ReadVariableOp_2,^LSTM_lay_0/while/lstm_cell/ReadVariableOp_30^LSTM_lay_0/while/lstm_cell/split/ReadVariableOp2^LSTM_lay_0/while/lstm_cell/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
LSTM_lay_0/while/Identity_3?
LSTM_lay_0/while/Identity_4Identity$LSTM_lay_0/while/lstm_cell/mul_5:z:0*^LSTM_lay_0/while/lstm_cell/ReadVariableOp,^LSTM_lay_0/while/lstm_cell/ReadVariableOp_1,^LSTM_lay_0/while/lstm_cell/ReadVariableOp_2,^LSTM_lay_0/while/lstm_cell/ReadVariableOp_30^LSTM_lay_0/while/lstm_cell/split/ReadVariableOp2^LSTM_lay_0/while/lstm_cell/split_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
LSTM_lay_0/while/Identity_4?
LSTM_lay_0/while/Identity_5Identity$LSTM_lay_0/while/lstm_cell/add_5:z:0*^LSTM_lay_0/while/lstm_cell/ReadVariableOp,^LSTM_lay_0/while/lstm_cell/ReadVariableOp_1,^LSTM_lay_0/while/lstm_cell/ReadVariableOp_2,^LSTM_lay_0/while/lstm_cell/ReadVariableOp_30^LSTM_lay_0/while/lstm_cell/split/ReadVariableOp2^LSTM_lay_0/while/lstm_cell/split_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
LSTM_lay_0/while/Identity_5"?
lstm_lay_0_while_identity"LSTM_lay_0/while/Identity:output:0"C
lstm_lay_0_while_identity_1$LSTM_lay_0/while/Identity_1:output:0"C
lstm_lay_0_while_identity_2$LSTM_lay_0/while/Identity_2:output:0"C
lstm_lay_0_while_identity_3$LSTM_lay_0/while/Identity_3:output:0"C
lstm_lay_0_while_identity_4$LSTM_lay_0/while/Identity_4:output:0"C
lstm_lay_0_while_identity_5$LSTM_lay_0/while/Identity_5:output:0"j
2lstm_lay_0_while_lstm_cell_readvariableop_resource4lstm_lay_0_while_lstm_cell_readvariableop_resource_0"z
:lstm_lay_0_while_lstm_cell_split_1_readvariableop_resource<lstm_lay_0_while_lstm_cell_split_1_readvariableop_resource_0"v
8lstm_lay_0_while_lstm_cell_split_readvariableop_resource:lstm_lay_0_while_lstm_cell_split_readvariableop_resource_0"\
+lstm_lay_0_while_lstm_lay_0_strided_slice_1-lstm_lay_0_while_lstm_lay_0_strided_slice_1_0"?
glstm_lay_0_while_tensorarrayv2read_tensorlistgetitem_lstm_lay_0_tensorarrayunstack_tensorlistfromtensorilstm_lay_0_while_tensorarrayv2read_tensorlistgetitem_lstm_lay_0_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :??????????:??????????: : :::2V
)LSTM_lay_0/while/lstm_cell/ReadVariableOp)LSTM_lay_0/while/lstm_cell/ReadVariableOp2Z
+LSTM_lay_0/while/lstm_cell/ReadVariableOp_1+LSTM_lay_0/while/lstm_cell/ReadVariableOp_12Z
+LSTM_lay_0/while/lstm_cell/ReadVariableOp_2+LSTM_lay_0/while/lstm_cell/ReadVariableOp_22Z
+LSTM_lay_0/while/lstm_cell/ReadVariableOp_3+LSTM_lay_0/while/lstm_cell/ReadVariableOp_32b
/LSTM_lay_0/while/lstm_cell/split/ReadVariableOp/LSTM_lay_0/while/lstm_cell/split/ReadVariableOp2f
1LSTM_lay_0/while/lstm_cell/split_1/ReadVariableOp1LSTM_lay_0/while/lstm_cell/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
*__inference_lstm_cell_layer_call_fn_122179

inputs
states_0
states_1
unknown
	unknown_0
	unknown_1
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_lstm_cell_layer_call_and_return_conditional_losses_1188112
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*[
_input_shapesJ
H:??????????:??????????:??????????:::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/1
?
?
while_cond_119324
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_119324___redundant_placeholder04
0while_while_cond_119324___redundant_placeholder14
0while_while_cond_119324___redundant_placeholder24
0while_while_cond_119324___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?V
?
E__inference_lstm_cell_layer_call_and_return_conditional_losses_118902

inputs

states
states_1!
split_readvariableop_resource#
split_1_readvariableop_resource
readvariableop_resource
identity

identity_1

identity_2??ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?ReadVariableOp_3?split/ReadVariableOp?split_1/ReadVariableOpP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource* 
_output_shapes
:
??*
dtype02
split/ReadVariableOp?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
??:
??:
??:
??*
	num_split2
splite
MatMulMatMulinputssplit:output:0*
T0*(
_output_shapes
:??????????2
MatMuli
MatMul_1MatMulinputssplit:output:1*
T0*(
_output_shapes
:??????????2

MatMul_1i
MatMul_2MatMulinputssplit:output:2*
T0*(
_output_shapes
:??????????2

MatMul_2i
MatMul_3MatMulinputssplit:output:3*
T0*(
_output_shapes
:??????????2

MatMul_3T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_1/split_dim?
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:?*
dtype02
split_1/ReadVariableOp?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2	
split_1t
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*(
_output_shapes
:??????????2	
BiasAddz
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*(
_output_shapes
:??????????2
	BiasAdd_1z
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*(
_output_shapes
:??????????2
	BiasAdd_2z
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*(
_output_shapes
:??????????2
	BiasAdd_3z
ReadVariableOpReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype02
ReadVariableOp{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
strided_sliceq
MatMul_4MatMulstatesstrided_slice:output:0*
T0*(
_output_shapes
:??????????2

MatMul_4l
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
addW
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_2W
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_3_
MulMuladd:z:0Const_2:output:0*
T0*(
_output_shapes
:??????????2
Mulc
Add_1AddMul:z:0Const_3:output:0*
T0*(
_output_shapes
:??????????2
Add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:??????????2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*(
_output_shapes
:??????????2
clip_by_value~
ReadVariableOp_1ReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype02
ReadVariableOp_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
strided_slice_1s
MatMul_5MatMulstatesstrided_slice_1:output:0*
T0*(
_output_shapes
:??????????2

MatMul_5r
add_2AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
add_2W
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_4W
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_5e
Mul_1Mul	add_2:z:0Const_4:output:0*
T0*(
_output_shapes
:??????????2
Mul_1e
Add_3Add	Mul_1:z:0Const_5:output:0*
T0*(
_output_shapes
:??????????2
Add_3{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_1/Minimum/y?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:??????????2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*(
_output_shapes
:??????????2
clip_by_value_1g
mul_2Mulclip_by_value_1:z:0states_1*
T0*(
_output_shapes
:??????????2
mul_2~
ReadVariableOp_2ReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype02
ReadVariableOp_2
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2
strided_slice_2/stack_1?
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2?
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
strided_slice_2s
MatMul_6MatMulstatesstrided_slice_2:output:0*
T0*(
_output_shapes
:??????????2

MatMul_6r
add_4AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
add_4[
SigmoidSigmoid	add_4:z:0*
T0*(
_output_shapes
:??????????2	
Sigmoidh
mul_3Mulclip_by_value:z:0Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
mul_3`
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*(
_output_shapes
:??????????2
add_5~
ReadVariableOp_3ReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype02
ReadVariableOp_3
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2
strided_slice_3/stack?
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack_1?
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2?
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
strided_slice_3s
MatMul_7MatMulstatesstrided_slice_3:output:0*
T0*(
_output_shapes
:??????????2

MatMul_7r
add_6AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
add_6W
Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_6W
Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_7e
Mul_4Mul	add_6:z:0Const_6:output:0*
T0*(
_output_shapes
:??????????2
Mul_4e
Add_7Add	Mul_4:z:0Const_7:output:0*
T0*(
_output_shapes
:??????????2
Add_7{
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_2/Minimum/y?
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:??????????2
clip_by_value_2/Minimumk
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_2/y?
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*(
_output_shapes
:??????????2
clip_by_value_2_
	Sigmoid_1Sigmoid	add_5:z:0*
T0*(
_output_shapes
:??????????2
	Sigmoid_1l
mul_5Mulclip_by_value_2:z:0Sigmoid_1:y:0*
T0*(
_output_shapes
:??????????2
mul_5?
IdentityIdentity	mul_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity	mul_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity_1?

Identity_2Identity	add_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*[
_input_shapesJ
H:??????????:??????????:??????????:::2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_namestates:PL
(
_output_shapes
:??????????
 
_user_specified_namestates
?
g
I__inference_Dropout_lay_0_layer_call_and_return_conditional_losses_121931

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
"__inference__traced_restore_122421
file_prefix'
#assignvariableop_dense_lay_1_kernel'
#assignvariableop_1_dense_lay_1_bias(
$assignvariableop_2_output_lay_kernel&
"assignvariableop_3_output_lay_bias 
assignvariableop_4_adam_iter"
assignvariableop_5_adam_beta_1"
assignvariableop_6_adam_beta_2!
assignvariableop_7_adam_decay)
%assignvariableop_8_adam_learning_rate2
.assignvariableop_9_lstm_lay_0_lstm_cell_kernel=
9assignvariableop_10_lstm_lay_0_lstm_cell_recurrent_kernel1
-assignvariableop_11_lstm_lay_0_lstm_cell_bias
assignvariableop_12_total
assignvariableop_13_count
assignvariableop_14_total_1
assignvariableop_15_count_1
assignvariableop_16_total_2
assignvariableop_17_count_21
-assignvariableop_18_adam_dense_lay_1_kernel_m/
+assignvariableop_19_adam_dense_lay_1_bias_m0
,assignvariableop_20_adam_output_lay_kernel_m.
*assignvariableop_21_adam_output_lay_bias_m:
6assignvariableop_22_adam_lstm_lay_0_lstm_cell_kernel_mD
@assignvariableop_23_adam_lstm_lay_0_lstm_cell_recurrent_kernel_m8
4assignvariableop_24_adam_lstm_lay_0_lstm_cell_bias_m1
-assignvariableop_25_adam_dense_lay_1_kernel_v/
+assignvariableop_26_adam_dense_lay_1_bias_v0
,assignvariableop_27_adam_output_lay_kernel_v.
*assignvariableop_28_adam_output_lay_bias_v:
6assignvariableop_29_adam_lstm_lay_0_lstm_cell_kernel_vD
@assignvariableop_30_adam_lstm_lay_0_lstm_cell_recurrent_kernel_v8
4assignvariableop_31_adam_lstm_lay_0_lstm_cell_bias_v
identity_33??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:!*
dtype0*?
value?B?!B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:!*
dtype0*U
valueLBJ!B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?:::::::::::::::::::::::::::::::::*/
dtypes%
#2!	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp#assignvariableop_dense_lay_1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp#assignvariableop_1_dense_lay_1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp$assignvariableop_2_output_lay_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp"assignvariableop_3_output_lay_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_iterIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_beta_1Identity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_beta_2Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_decayIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp%assignvariableop_8_adam_learning_rateIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp.assignvariableop_9_lstm_lay_0_lstm_cell_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp9assignvariableop_10_lstm_lay_0_lstm_cell_recurrent_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp-assignvariableop_11_lstm_lay_0_lstm_cell_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOpassignvariableop_12_totalIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpassignvariableop_13_countIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOpassignvariableop_14_total_1Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpassignvariableop_15_count_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOpassignvariableop_16_total_2Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOpassignvariableop_17_count_2Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp-assignvariableop_18_adam_dense_lay_1_kernel_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp+assignvariableop_19_adam_dense_lay_1_bias_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp,assignvariableop_20_adam_output_lay_kernel_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp*assignvariableop_21_adam_output_lay_bias_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp6assignvariableop_22_adam_lstm_lay_0_lstm_cell_kernel_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp@assignvariableop_23_adam_lstm_lay_0_lstm_cell_recurrent_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp4assignvariableop_24_adam_lstm_lay_0_lstm_cell_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp-assignvariableop_25_adam_dense_lay_1_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp+assignvariableop_26_adam_dense_lay_1_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp,assignvariableop_27_adam_output_lay_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp*assignvariableop_28_adam_output_lay_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp6assignvariableop_29_adam_lstm_lay_0_lstm_cell_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp@assignvariableop_30_adam_lstm_lay_0_lstm_cell_recurrent_kernel_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp4assignvariableop_31_adam_lstm_lay_0_lstm_cell_bias_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_319
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_32Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_32?
Identity_33IdentityIdentity_32:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_33"#
identity_33Identity_33:output:0*?
_input_shapes?
?: ::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
while_cond_119530
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_119530___redundant_placeholder04
0while_while_cond_119530___redundant_placeholder14
0while_while_cond_119530___redundant_placeholder24
0while_while_cond_119530___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
,__inference_Dense_lay_1_layer_call_fn_121961

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_Dense_lay_1_layer_call_and_return_conditional_losses_1200122
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
g
I__inference_Dropout_lay_0_layer_call_and_return_conditional_losses_119988

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
while_body_120924
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_03
/while_lstm_cell_split_readvariableop_resource_05
1while_lstm_cell_split_1_readvariableop_resource_0-
)while_lstm_cell_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor1
-while_lstm_cell_split_readvariableop_resource3
/while_lstm_cell_split_1_readvariableop_resource+
'while_lstm_cell_readvariableop_resource??while/lstm_cell/ReadVariableOp? while/lstm_cell/ReadVariableOp_1? while/lstm_cell/ReadVariableOp_2? while/lstm_cell/ReadVariableOp_3?$while/lstm_cell/split/ReadVariableOp?&while/lstm_cell/split_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemp
while/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell/Const?
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2!
while/lstm_cell/split/split_dim?
$while/lstm_cell/split/ReadVariableOpReadVariableOp/while_lstm_cell_split_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02&
$while/lstm_cell/split/ReadVariableOp?
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0,while/lstm_cell/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
??:
??:
??:
??*
	num_split2
while/lstm_cell/split?
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/MatMul?
while/lstm_cell/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:1*
T0*(
_output_shapes
:??????????2
while/lstm_cell/MatMul_1?
while/lstm_cell/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:2*
T0*(
_output_shapes
:??????????2
while/lstm_cell/MatMul_2?
while/lstm_cell/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:3*
T0*(
_output_shapes
:??????????2
while/lstm_cell/MatMul_3t
while/lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell/Const_1?
!while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!while/lstm_cell/split_1/split_dim?
&while/lstm_cell/split_1/ReadVariableOpReadVariableOp1while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
:?*
dtype02(
&while/lstm_cell/split_1/ReadVariableOp?
while/lstm_cell/split_1Split*while/lstm_cell/split_1/split_dim:output:0.while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2
while/lstm_cell/split_1?
while/lstm_cell/BiasAddBiasAdd while/lstm_cell/MatMul:product:0 while/lstm_cell/split_1:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/BiasAdd?
while/lstm_cell/BiasAdd_1BiasAdd"while/lstm_cell/MatMul_1:product:0 while/lstm_cell/split_1:output:1*
T0*(
_output_shapes
:??????????2
while/lstm_cell/BiasAdd_1?
while/lstm_cell/BiasAdd_2BiasAdd"while/lstm_cell/MatMul_2:product:0 while/lstm_cell/split_1:output:2*
T0*(
_output_shapes
:??????????2
while/lstm_cell/BiasAdd_2?
while/lstm_cell/BiasAdd_3BiasAdd"while/lstm_cell/MatMul_3:product:0 while/lstm_cell/split_1:output:3*
T0*(
_output_shapes
:??????????2
while/lstm_cell/BiasAdd_3?
while/lstm_cell/ReadVariableOpReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02 
while/lstm_cell/ReadVariableOp?
#while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2%
#while/lstm_cell/strided_slice/stack?
%while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2'
%while/lstm_cell/strided_slice/stack_1?
%while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2'
%while/lstm_cell/strided_slice/stack_2?
while/lstm_cell/strided_sliceStridedSlice&while/lstm_cell/ReadVariableOp:value:0,while/lstm_cell/strided_slice/stack:output:0.while/lstm_cell/strided_slice/stack_1:output:0.while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
while/lstm_cell/strided_slice?
while/lstm_cell/MatMul_4MatMulwhile_placeholder_2&while/lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/MatMul_4?
while/lstm_cell/addAddV2 while/lstm_cell/BiasAdd:output:0"while/lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/addw
while/lstm_cell/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/lstm_cell/Const_2w
while/lstm_cell/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/lstm_cell/Const_3?
while/lstm_cell/MulMulwhile/lstm_cell/add:z:0 while/lstm_cell/Const_2:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/Mul?
while/lstm_cell/Add_1Addwhile/lstm_cell/Mul:z:0 while/lstm_cell/Const_3:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/Add_1?
'while/lstm_cell/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2)
'while/lstm_cell/clip_by_value/Minimum/y?
%while/lstm_cell/clip_by_value/MinimumMinimumwhile/lstm_cell/Add_1:z:00while/lstm_cell/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:??????????2'
%while/lstm_cell/clip_by_value/Minimum?
while/lstm_cell/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
while/lstm_cell/clip_by_value/y?
while/lstm_cell/clip_by_valueMaximum)while/lstm_cell/clip_by_value/Minimum:z:0(while/lstm_cell/clip_by_value/y:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/clip_by_value?
 while/lstm_cell/ReadVariableOp_1ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02"
 while/lstm_cell/ReadVariableOp_1?
%while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2'
%while/lstm_cell/strided_slice_1/stack?
'while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2)
'while/lstm_cell/strided_slice_1/stack_1?
'while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_1/stack_2?
while/lstm_cell/strided_slice_1StridedSlice(while/lstm_cell/ReadVariableOp_1:value:0.while/lstm_cell/strided_slice_1/stack:output:00while/lstm_cell/strided_slice_1/stack_1:output:00while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2!
while/lstm_cell/strided_slice_1?
while/lstm_cell/MatMul_5MatMulwhile_placeholder_2(while/lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/MatMul_5?
while/lstm_cell/add_2AddV2"while/lstm_cell/BiasAdd_1:output:0"while/lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/add_2w
while/lstm_cell/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/lstm_cell/Const_4w
while/lstm_cell/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/lstm_cell/Const_5?
while/lstm_cell/Mul_1Mulwhile/lstm_cell/add_2:z:0 while/lstm_cell/Const_4:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/Mul_1?
while/lstm_cell/Add_3Addwhile/lstm_cell/Mul_1:z:0 while/lstm_cell/Const_5:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/Add_3?
)while/lstm_cell/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2+
)while/lstm_cell/clip_by_value_1/Minimum/y?
'while/lstm_cell/clip_by_value_1/MinimumMinimumwhile/lstm_cell/Add_3:z:02while/lstm_cell/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:??????????2)
'while/lstm_cell/clip_by_value_1/Minimum?
!while/lstm_cell/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!while/lstm_cell/clip_by_value_1/y?
while/lstm_cell/clip_by_value_1Maximum+while/lstm_cell/clip_by_value_1/Minimum:z:0*while/lstm_cell/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:??????????2!
while/lstm_cell/clip_by_value_1?
while/lstm_cell/mul_2Mul#while/lstm_cell/clip_by_value_1:z:0while_placeholder_3*
T0*(
_output_shapes
:??????????2
while/lstm_cell/mul_2?
 while/lstm_cell/ReadVariableOp_2ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02"
 while/lstm_cell/ReadVariableOp_2?
%while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2'
%while/lstm_cell/strided_slice_2/stack?
'while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2)
'while/lstm_cell/strided_slice_2/stack_1?
'while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_2/stack_2?
while/lstm_cell/strided_slice_2StridedSlice(while/lstm_cell/ReadVariableOp_2:value:0.while/lstm_cell/strided_slice_2/stack:output:00while/lstm_cell/strided_slice_2/stack_1:output:00while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2!
while/lstm_cell/strided_slice_2?
while/lstm_cell/MatMul_6MatMulwhile_placeholder_2(while/lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/MatMul_6?
while/lstm_cell/add_4AddV2"while/lstm_cell/BiasAdd_2:output:0"while/lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/add_4?
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/add_4:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/Sigmoid?
while/lstm_cell/mul_3Mul!while/lstm_cell/clip_by_value:z:0while/lstm_cell/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/mul_3?
while/lstm_cell/add_5AddV2while/lstm_cell/mul_2:z:0while/lstm_cell/mul_3:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/add_5?
 while/lstm_cell/ReadVariableOp_3ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02"
 while/lstm_cell/ReadVariableOp_3?
%while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2'
%while/lstm_cell/strided_slice_3/stack?
'while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'while/lstm_cell/strided_slice_3/stack_1?
'while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_3/stack_2?
while/lstm_cell/strided_slice_3StridedSlice(while/lstm_cell/ReadVariableOp_3:value:0.while/lstm_cell/strided_slice_3/stack:output:00while/lstm_cell/strided_slice_3/stack_1:output:00while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2!
while/lstm_cell/strided_slice_3?
while/lstm_cell/MatMul_7MatMulwhile_placeholder_2(while/lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/MatMul_7?
while/lstm_cell/add_6AddV2"while/lstm_cell/BiasAdd_3:output:0"while/lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/add_6w
while/lstm_cell/Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/lstm_cell/Const_6w
while/lstm_cell/Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/lstm_cell/Const_7?
while/lstm_cell/Mul_4Mulwhile/lstm_cell/add_6:z:0 while/lstm_cell/Const_6:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/Mul_4?
while/lstm_cell/Add_7Addwhile/lstm_cell/Mul_4:z:0 while/lstm_cell/Const_7:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/Add_7?
)while/lstm_cell/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2+
)while/lstm_cell/clip_by_value_2/Minimum/y?
'while/lstm_cell/clip_by_value_2/MinimumMinimumwhile/lstm_cell/Add_7:z:02while/lstm_cell/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:??????????2)
'while/lstm_cell/clip_by_value_2/Minimum?
!while/lstm_cell/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!while/lstm_cell/clip_by_value_2/y?
while/lstm_cell/clip_by_value_2Maximum+while/lstm_cell/clip_by_value_2/Minimum:z:0*while/lstm_cell/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:??????????2!
while/lstm_cell/clip_by_value_2?
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/add_5:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/Sigmoid_1?
while/lstm_cell/mul_5Mul#while/lstm_cell/clip_by_value_2:z:0while/lstm_cell/Sigmoid_1:y:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell/mul_5?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_5:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:0^while/lstm_cell/ReadVariableOp!^while/lstm_cell/ReadVariableOp_1!^while/lstm_cell/ReadVariableOp_2!^while/lstm_cell/ReadVariableOp_3%^while/lstm_cell/split/ReadVariableOp'^while/lstm_cell/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations^while/lstm_cell/ReadVariableOp!^while/lstm_cell/ReadVariableOp_1!^while/lstm_cell/ReadVariableOp_2!^while/lstm_cell/ReadVariableOp_3%^while/lstm_cell/split/ReadVariableOp'^while/lstm_cell/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0^while/lstm_cell/ReadVariableOp!^while/lstm_cell/ReadVariableOp_1!^while/lstm_cell/ReadVariableOp_2!^while/lstm_cell/ReadVariableOp_3%^while/lstm_cell/split/ReadVariableOp'^while/lstm_cell/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/lstm_cell/ReadVariableOp!^while/lstm_cell/ReadVariableOp_1!^while/lstm_cell/ReadVariableOp_2!^while/lstm_cell/ReadVariableOp_3%^while/lstm_cell/split/ReadVariableOp'^while/lstm_cell/split_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell/mul_5:z:0^while/lstm_cell/ReadVariableOp!^while/lstm_cell/ReadVariableOp_1!^while/lstm_cell/ReadVariableOp_2!^while/lstm_cell/ReadVariableOp_3%^while/lstm_cell/split/ReadVariableOp'^while/lstm_cell/split_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell/add_5:z:0^while/lstm_cell/ReadVariableOp!^while/lstm_cell/ReadVariableOp_1!^while/lstm_cell/ReadVariableOp_2!^while/lstm_cell/ReadVariableOp_3%^while/lstm_cell/split/ReadVariableOp'^while/lstm_cell/split_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"T
'while_lstm_cell_readvariableop_resource)while_lstm_cell_readvariableop_resource_0"d
/while_lstm_cell_split_1_readvariableop_resource1while_lstm_cell_split_1_readvariableop_resource_0"`
-while_lstm_cell_split_readvariableop_resource/while_lstm_cell_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :??????????:??????????: : :::2@
while/lstm_cell/ReadVariableOpwhile/lstm_cell/ReadVariableOp2D
 while/lstm_cell/ReadVariableOp_1 while/lstm_cell/ReadVariableOp_12D
 while/lstm_cell/ReadVariableOp_2 while/lstm_cell/ReadVariableOp_22D
 while/lstm_cell/ReadVariableOp_3 while/lstm_cell/ReadVariableOp_32L
$while/lstm_cell/split/ReadVariableOp$while/lstm_cell/split/ReadVariableOp2P
&while/lstm_cell/split_1/ReadVariableOp&while/lstm_cell/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
J
.__inference_Dropout_lay_0_layer_call_fn_121941

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_Dropout_lay_0_layer_call_and_return_conditional_losses_1199882
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?V
?
E__inference_lstm_cell_layer_call_and_return_conditional_losses_122162

inputs
states_0
states_1!
split_readvariableop_resource#
split_1_readvariableop_resource
readvariableop_resource
identity

identity_1

identity_2??ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?ReadVariableOp_3?split/ReadVariableOp?split_1/ReadVariableOpP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource* 
_output_shapes
:
??*
dtype02
split/ReadVariableOp?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
??:
??:
??:
??*
	num_split2
splite
MatMulMatMulinputssplit:output:0*
T0*(
_output_shapes
:??????????2
MatMuli
MatMul_1MatMulinputssplit:output:1*
T0*(
_output_shapes
:??????????2

MatMul_1i
MatMul_2MatMulinputssplit:output:2*
T0*(
_output_shapes
:??????????2

MatMul_2i
MatMul_3MatMulinputssplit:output:3*
T0*(
_output_shapes
:??????????2

MatMul_3T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_1/split_dim?
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:?*
dtype02
split_1/ReadVariableOp?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?:?:?:?*
	num_split2	
split_1t
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*(
_output_shapes
:??????????2	
BiasAddz
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*(
_output_shapes
:??????????2
	BiasAdd_1z
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*(
_output_shapes
:??????????2
	BiasAdd_2z
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*(
_output_shapes
:??????????2
	BiasAdd_3z
ReadVariableOpReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype02
ReadVariableOp{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?   2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
strided_slices
MatMul_4MatMulstates_0strided_slice:output:0*
T0*(
_output_shapes
:??????????2

MatMul_4l
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*(
_output_shapes
:??????????2
addW
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_2W
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_3_
MulMuladd:z:0Const_2:output:0*
T0*(
_output_shapes
:??????????2
Mulc
Add_1AddMul:z:0Const_3:output:0*
T0*(
_output_shapes
:??????????2
Add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:??????????2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*(
_output_shapes
:??????????2
clip_by_value~
ReadVariableOp_1ReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype02
ReadVariableOp_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?   2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
strided_slice_1u
MatMul_5MatMulstates_0strided_slice_1:output:0*
T0*(
_output_shapes
:??????????2

MatMul_5r
add_2AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*(
_output_shapes
:??????????2
add_2W
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_4W
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_5e
Mul_1Mul	add_2:z:0Const_4:output:0*
T0*(
_output_shapes
:??????????2
Mul_1e
Add_3Add	Mul_1:z:0Const_5:output:0*
T0*(
_output_shapes
:??????????2
Add_3{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_1/Minimum/y?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:??????????2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*(
_output_shapes
:??????????2
clip_by_value_1g
mul_2Mulclip_by_value_1:z:0states_1*
T0*(
_output_shapes
:??????????2
mul_2~
ReadVariableOp_2ReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype02
ReadVariableOp_2
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2
strided_slice_2/stack_1?
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2?
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
strided_slice_2u
MatMul_6MatMulstates_0strided_slice_2:output:0*
T0*(
_output_shapes
:??????????2

MatMul_6r
add_4AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*(
_output_shapes
:??????????2
add_4[
SigmoidSigmoid	add_4:z:0*
T0*(
_output_shapes
:??????????2	
Sigmoidh
mul_3Mulclip_by_value:z:0Sigmoid:y:0*
T0*(
_output_shapes
:??????????2
mul_3`
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*(
_output_shapes
:??????????2
add_5~
ReadVariableOp_3ReadVariableOpreadvariableop_resource* 
_output_shapes
:
??*
dtype02
ReadVariableOp_3
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2
strided_slice_3/stack?
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack_1?
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2?
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
??*

begin_mask*
end_mask2
strided_slice_3u
MatMul_7MatMulstates_0strided_slice_3:output:0*
T0*(
_output_shapes
:??????????2

MatMul_7r
add_6AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*(
_output_shapes
:??????????2
add_6W
Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_6W
Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_7e
Mul_4Mul	add_6:z:0Const_6:output:0*
T0*(
_output_shapes
:??????????2
Mul_4e
Add_7Add	Mul_4:z:0Const_7:output:0*
T0*(
_output_shapes
:??????????2
Add_7{
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_2/Minimum/y?
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:??????????2
clip_by_value_2/Minimumk
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_2/y?
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*(
_output_shapes
:??????????2
clip_by_value_2_
	Sigmoid_1Sigmoid	add_5:z:0*
T0*(
_output_shapes
:??????????2
	Sigmoid_1l
mul_5Mulclip_by_value_2:z:0Sigmoid_1:y:0*
T0*(
_output_shapes
:??????????2
mul_5?
IdentityIdentity	mul_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity	mul_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity_1?

Identity_2Identity	add_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*[
_input_shapesJ
H:??????????:??????????:??????????:::2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/1
?C
?
F__inference_LSTM_lay_0_layer_call_and_return_conditional_losses_119262

inputs
lstm_cell_119181
lstm_cell_119183
lstm_cell_119185
identity??!lstm_cell/StatefulPartitionedCall?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_2?
!lstm_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_119181lstm_cell_119183lstm_cell_119185*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_lstm_cell_layer_call_and_return_conditional_losses_1188112#
!lstm_cell/StatefulPartitionedCall?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_119181lstm_cell_119183lstm_cell_119185*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_119194*
condR
while_cond_119193*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
transpose_1?
IdentityIdentitystrided_slice_3:output:0"^lstm_cell/StatefulPartitionedCall^while*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:???????????????????:::2F
!lstm_cell/StatefulPartitionedCall!lstm_cell/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?
?
+__inference_sequential_layer_call_fn_120798

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????`*)
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_1201432
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????`2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:??????????:::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
R
LSTM_lay_0_input>
"serving_default_LSTM_lay_0_input:0??????????>

Output_lay0
StatefulPartitionedCall:0?????????`tensorflow/serving/predict:??
?-
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
	optimizer
trainable_variables
	variables
regularization_losses
		keras_api


signatures
k__call__
*l&call_and_return_all_conditional_losses
m_default_save_signature"?+
_tf_keras_sequential?*{"class_name": "Sequential", "name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 200]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "LSTM_lay_0_input"}}, {"class_name": "LSTM", "config": {"name": "LSTM_lay_0", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 200]}, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 128, "activation": "sigmoid", "recurrent_activation": "hard_sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 1}}, {"class_name": "Dropout", "config": {"name": "Dropout_lay_0", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "Dense_lay_1", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "Output_lay", "trainable": true, "dtype": "float32", "units": 96, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 200]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 1, 200]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 200]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "LSTM_lay_0_input"}}, {"class_name": "LSTM", "config": {"name": "LSTM_lay_0", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 200]}, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 128, "activation": "sigmoid", "recurrent_activation": "hard_sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 1}}, {"class_name": "Dropout", "config": {"name": "Dropout_lay_0", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "Dense_lay_1", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "Output_lay", "trainable": true, "dtype": "float32", "units": 96, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "mse", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "mape", "dtype": "float32", "fn": "mean_absolute_percentage_error"}}, {"class_name": "MeanMetricWrapper", "config": {"name": "mse", "dtype": "float32", "fn": "mean_squared_error"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 9.999999747378752e-05, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?
cell

state_spec
trainable_variables
	variables
regularization_losses
	keras_api
n__call__
*o&call_and_return_all_conditional_losses"?
_tf_keras_rnn_layer?{"class_name": "LSTM", "name": "LSTM_lay_0", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 200]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "LSTM_lay_0", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 200]}, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 128, "activation": "sigmoid", "recurrent_activation": "hard_sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 1}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 200]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 1, 200]}}
?
trainable_variables
	variables
regularization_losses
	keras_api
p__call__
*q&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dropout", "name": "Dropout_lay_0", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Dropout_lay_0", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
?

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
r__call__
*s&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "Dense_lay_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Dense_lay_1", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
?

kernel
bias
trainable_variables
	variables
regularization_losses
 	keras_api
t__call__
*u&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "Output_lay", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Output_lay", "trainable": true, "dtype": "float32", "units": 96, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
?
!iter

"beta_1

#beta_2
	$decay
%learning_ratem]m^m_m`&ma'mb(mcvdvevfvg&vh'vi(vj"
	optimizer
Q
&0
'1
(2
3
4
5
6"
trackable_list_wrapper
Q
&0
'1
(2
3
4
5
6"
trackable_list_wrapper
 "
trackable_list_wrapper
?
trainable_variables
)layer_regularization_losses
*metrics
+non_trainable_variables

,layers
	variables
-layer_metrics
regularization_losses
k__call__
m_default_save_signature
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses"
_generic_user_object
,
vserving_default"
signature_map
?

&kernel
'recurrent_kernel
(bias
.trainable_variables
/	variables
0regularization_losses
1	keras_api
w__call__
*x&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "LSTMCell", "name": "lstm_cell", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm_cell", "trainable": true, "dtype": "float32", "units": 128, "activation": "sigmoid", "recurrent_activation": "hard_sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 1}}
 "
trackable_list_wrapper
5
&0
'1
(2"
trackable_list_wrapper
5
&0
'1
(2"
trackable_list_wrapper
 "
trackable_list_wrapper
?
trainable_variables
2layer_regularization_losses
3metrics
4non_trainable_variables

5layers
	variables
6layer_metrics
regularization_losses

7states
n__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
trainable_variables
	variables
8layer_regularization_losses
9metrics

:layers
;non_trainable_variables
<layer_metrics
regularization_losses
p__call__
*q&call_and_return_all_conditional_losses
&q"call_and_return_conditional_losses"
_generic_user_object
&:$
??2Dense_lay_1/kernel
:?2Dense_lay_1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
trainable_variables
	variables
=layer_regularization_losses
>metrics

?layers
@non_trainable_variables
Alayer_metrics
regularization_losses
r__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses"
_generic_user_object
$:"	?`2Output_lay/kernel
:`2Output_lay/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
trainable_variables
	variables
Blayer_regularization_losses
Cmetrics

Dlayers
Enon_trainable_variables
Flayer_metrics
regularization_losses
t__call__
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
/:-
??2LSTM_lay_0/lstm_cell/kernel
9:7
??2%LSTM_lay_0/lstm_cell/recurrent_kernel
(:&?2LSTM_lay_0/lstm_cell/bias
 "
trackable_list_wrapper
5
G0
H1
I2"
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_dict_wrapper
5
&0
'1
(2"
trackable_list_wrapper
5
&0
'1
(2"
trackable_list_wrapper
 "
trackable_list_wrapper
?
.trainable_variables
/	variables
Jlayer_regularization_losses
Kmetrics

Llayers
Mnon_trainable_variables
Nlayer_metrics
0regularization_losses
w__call__
*x&call_and_return_all_conditional_losses
&x"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?
	Ototal
	Pcount
Q	variables
R	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
?
	Stotal
	Tcount
U
_fn_kwargs
V	variables
W	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "mape", "dtype": "float32", "config": {"name": "mape", "dtype": "float32", "fn": "mean_absolute_percentage_error"}}
?
	Xtotal
	Ycount
Z
_fn_kwargs
[	variables
\	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "mse", "dtype": "float32", "config": {"name": "mse", "dtype": "float32", "fn": "mean_squared_error"}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
:  (2total
:  (2count
.
O0
P1"
trackable_list_wrapper
-
Q	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
S0
T1"
trackable_list_wrapper
-
V	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
X0
Y1"
trackable_list_wrapper
-
[	variables"
_generic_user_object
+:)
??2Adam/Dense_lay_1/kernel/m
$:"?2Adam/Dense_lay_1/bias/m
):'	?`2Adam/Output_lay/kernel/m
": `2Adam/Output_lay/bias/m
4:2
??2"Adam/LSTM_lay_0/lstm_cell/kernel/m
>:<
??2,Adam/LSTM_lay_0/lstm_cell/recurrent_kernel/m
-:+?2 Adam/LSTM_lay_0/lstm_cell/bias/m
+:)
??2Adam/Dense_lay_1/kernel/v
$:"?2Adam/Dense_lay_1/bias/v
):'	?`2Adam/Output_lay/kernel/v
": `2Adam/Output_lay/bias/v
4:2
??2"Adam/LSTM_lay_0/lstm_cell/kernel/v
>:<
??2,Adam/LSTM_lay_0/lstm_cell/recurrent_kernel/v
-:+?2 Adam/LSTM_lay_0/lstm_cell/bias/v
?2?
+__inference_sequential_layer_call_fn_120779
+__inference_sequential_layer_call_fn_120798
+__inference_sequential_layer_call_fn_120119
+__inference_sequential_layer_call_fn_120160?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
F__inference_sequential_layer_call_and_return_conditional_losses_120760
F__inference_sequential_layer_call_and_return_conditional_losses_120077
F__inference_sequential_layer_call_and_return_conditional_losses_120055
F__inference_sequential_layer_call_and_return_conditional_losses_120478?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
!__inference__wrapped_model_118680?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *4?1
/?,
LSTM_lay_0_input??????????
?2?
+__inference_LSTM_lay_0_layer_call_fn_121356
+__inference_LSTM_lay_0_layer_call_fn_121914
+__inference_LSTM_lay_0_layer_call_fn_121903
+__inference_LSTM_lay_0_layer_call_fn_121345?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
F__inference_LSTM_lay_0_layer_call_and_return_conditional_losses_121624
F__inference_LSTM_lay_0_layer_call_and_return_conditional_losses_121066
F__inference_LSTM_lay_0_layer_call_and_return_conditional_losses_121334
F__inference_LSTM_lay_0_layer_call_and_return_conditional_losses_121892?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
.__inference_Dropout_lay_0_layer_call_fn_121936
.__inference_Dropout_lay_0_layer_call_fn_121941?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
I__inference_Dropout_lay_0_layer_call_and_return_conditional_losses_121926
I__inference_Dropout_lay_0_layer_call_and_return_conditional_losses_121931?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
,__inference_Dense_lay_1_layer_call_fn_121961?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_Dense_lay_1_layer_call_and_return_conditional_losses_121952?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_Output_lay_layer_call_fn_121980?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_Output_lay_layer_call_and_return_conditional_losses_121971?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
$__inference_signature_wrapper_120189LSTM_lay_0_input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_lstm_cell_layer_call_fn_122179
*__inference_lstm_cell_layer_call_fn_122196?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_lstm_cell_layer_call_and_return_conditional_losses_122162
E__inference_lstm_cell_layer_call_and_return_conditional_losses_122071?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 ?
G__inference_Dense_lay_1_layer_call_and_return_conditional_losses_121952^0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
,__inference_Dense_lay_1_layer_call_fn_121961Q0?-
&?#
!?
inputs??????????
? "????????????
I__inference_Dropout_lay_0_layer_call_and_return_conditional_losses_121926^4?1
*?'
!?
inputs??????????
p
? "&?#
?
0??????????
? ?
I__inference_Dropout_lay_0_layer_call_and_return_conditional_losses_121931^4?1
*?'
!?
inputs??????????
p 
? "&?#
?
0??????????
? ?
.__inference_Dropout_lay_0_layer_call_fn_121936Q4?1
*?'
!?
inputs??????????
p
? "????????????
.__inference_Dropout_lay_0_layer_call_fn_121941Q4?1
*?'
!?
inputs??????????
p 
? "????????????
F__inference_LSTM_lay_0_layer_call_and_return_conditional_losses_121066o&('@?=
6?3
%?"
inputs??????????

 
p

 
? "&?#
?
0??????????
? ?
F__inference_LSTM_lay_0_layer_call_and_return_conditional_losses_121334o&('@?=
6?3
%?"
inputs??????????

 
p 

 
? "&?#
?
0??????????
? ?
F__inference_LSTM_lay_0_layer_call_and_return_conditional_losses_121624&('P?M
F?C
5?2
0?-
inputs/0???????????????????

 
p

 
? "&?#
?
0??????????
? ?
F__inference_LSTM_lay_0_layer_call_and_return_conditional_losses_121892&('P?M
F?C
5?2
0?-
inputs/0???????????????????

 
p 

 
? "&?#
?
0??????????
? ?
+__inference_LSTM_lay_0_layer_call_fn_121345b&('@?=
6?3
%?"
inputs??????????

 
p

 
? "????????????
+__inference_LSTM_lay_0_layer_call_fn_121356b&('@?=
6?3
%?"
inputs??????????

 
p 

 
? "????????????
+__inference_LSTM_lay_0_layer_call_fn_121903r&('P?M
F?C
5?2
0?-
inputs/0???????????????????

 
p

 
? "????????????
+__inference_LSTM_lay_0_layer_call_fn_121914r&('P?M
F?C
5?2
0?-
inputs/0???????????????????

 
p 

 
? "????????????
F__inference_Output_lay_layer_call_and_return_conditional_losses_121971]0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????`
? 
+__inference_Output_lay_layer_call_fn_121980P0?-
&?#
!?
inputs??????????
? "??????????`?
!__inference__wrapped_model_118680?&('>?;
4?1
/?,
LSTM_lay_0_input??????????
? "7?4
2

Output_lay$?!

Output_lay?????????`?
E__inference_lstm_cell_layer_call_and_return_conditional_losses_122071?&('???
y?v
!?
inputs??????????
M?J
#? 
states/0??????????
#? 
states/1??????????
p
? "v?s
l?i
?
0/0??????????
G?D
 ?
0/1/0??????????
 ?
0/1/1??????????
? ?
E__inference_lstm_cell_layer_call_and_return_conditional_losses_122162?&('???
y?v
!?
inputs??????????
M?J
#? 
states/0??????????
#? 
states/1??????????
p 
? "v?s
l?i
?
0/0??????????
G?D
 ?
0/1/0??????????
 ?
0/1/1??????????
? ?
*__inference_lstm_cell_layer_call_fn_122179?&('???
y?v
!?
inputs??????????
M?J
#? 
states/0??????????
#? 
states/1??????????
p
? "f?c
?
0??????????
C?@
?
1/0??????????
?
1/1???????????
*__inference_lstm_cell_layer_call_fn_122196?&('???
y?v
!?
inputs??????????
M?J
#? 
states/0??????????
#? 
states/1??????????
p 
? "f?c
?
0??????????
C?@
?
1/0??????????
?
1/1???????????
F__inference_sequential_layer_call_and_return_conditional_losses_120055x&('F?C
<?9
/?,
LSTM_lay_0_input??????????
p

 
? "%?"
?
0?????????`
? ?
F__inference_sequential_layer_call_and_return_conditional_losses_120077x&('F?C
<?9
/?,
LSTM_lay_0_input??????????
p 

 
? "%?"
?
0?????????`
? ?
F__inference_sequential_layer_call_and_return_conditional_losses_120478n&('<?9
2?/
%?"
inputs??????????
p

 
? "%?"
?
0?????????`
? ?
F__inference_sequential_layer_call_and_return_conditional_losses_120760n&('<?9
2?/
%?"
inputs??????????
p 

 
? "%?"
?
0?????????`
? ?
+__inference_sequential_layer_call_fn_120119k&('F?C
<?9
/?,
LSTM_lay_0_input??????????
p

 
? "??????????`?
+__inference_sequential_layer_call_fn_120160k&('F?C
<?9
/?,
LSTM_lay_0_input??????????
p 

 
? "??????????`?
+__inference_sequential_layer_call_fn_120779a&('<?9
2?/
%?"
inputs??????????
p

 
? "??????????`?
+__inference_sequential_layer_call_fn_120798a&('<?9
2?/
%?"
inputs??????????
p 

 
? "??????????`?
$__inference_signature_wrapper_120189?&('R?O
? 
H?E
C
LSTM_lay_0_input/?,
LSTM_lay_0_input??????????"7?4
2

Output_lay$?!

Output_lay?????????`