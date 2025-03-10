# Copyright 2020 InterDigital Communications, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from compressai.models import SymmetricalTransFormer, WACNN, STFFullOptimizer, STFCompressOptimizer, STFDecompressOptimizer, STFDemonstrateNoQuantization

from .pretrained import load_pretrained as load_state_dict

models = {
    'stf': SymmetricalTransFormer,
    'cnn': WACNN,
    'stf_full_optimizer': STFFullOptimizer,
    'stf_compress_optimizer': STFCompressOptimizer,
    'stf_decompress_optimizer': STFDecompressOptimizer,
    'stf_demonstrate_no_quantization': STFDemonstrateNoQuantization,
}
