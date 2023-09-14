var searchIndex = JSON.parse('{\
"rust_cuda":{"doc":"","t":"OOOAAAOOAAQIQDDDYQIILLLKKLLLLLLLLLLLLLLKKKLLLXLLLKLLLLLLLLLLLLLIDLLLLLLLLLLAKKKDDDLLFFLLLLLLLLLLLLFFLLLLFLLLLLLLLLMMMMMMNDQIDIDDDDEQDDIIDNDLLLMLLLLLLLLLLLLLLLLLLLLLLLLLLLMLLLLLLLLLLLLLLLLLLLLLLLLLLLLLKMLLLLLLLLLLLMKKKLLLLLMMLMLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLMLLLIIIIIAAAADDLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLDLLLLLLLLLDLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLAADDDDLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLDDLLLLLLLLLLLLLLLLLLLLL","n":["assert","assert_eq","assert_ne","common","device","host","print","println","safety","utils","CudaAllocation","CudaAsRust","CudaRepresentation","DeviceAccessible","DeviceConstRef","DeviceMutRef","LendRustToCuda","RustRepresentation","RustToCuda","RustToCudaProxy","as_mut","as_ref","as_ref","as_rust","borrow","borrow","borrow","borrow","borrow_mut","borrow_mut","borrow_mut","clone","clone_into","fmt","from","from","from","from","from","from_mut","from_ref","into","into","into","into","kernel","populate_graph","populate_graph","populate_graph","restore","to_owned","try_from","try_from","try_from","try_into","try_into","try_into","type_id","type_id","type_id","uninit","uninit","uninit","BorrowFromRust","ShallowCopy","borrow","borrow_mut","deref","deref_mut","fmt","from","into","try_from","try_into","type_id","utils","with_borrow_from_rust","with_borrow_from_rust_mut","with_moved_from_rust","Dim3","Idx3","PTXAllocator","alloc","as_id","block_dim","block_idx","borrow","borrow","borrow","borrow_mut","borrow_mut","borrow_mut","dealloc","fmt","fmt","from","from","from","grid_dim","index","into","into","into","size","thread_idx","try_from","try_from","try_from","try_into","try_into","try_into","type_id","type_id","type_id","x","x","y","y","z","z","Cached","CombinedCudaAlloc","CompilationWatcher","CudaAlloc","CudaDropWrapper","EmptyCudaAlloc","HostAndDeviceConstRef","HostAndDeviceMutRef","HostAndDeviceOwned","HostDeviceBox","KernelJITResult","KernelTraitObject","LaunchConfig","LaunchPackage","Launcher","LendToCuda","NullCudaAlloc","Recompiled","TypedKernel","as_mut","as_ref","as_ref","block","borrow","borrow","borrow","borrow","borrow","borrow","borrow","borrow","borrow","borrow","borrow","borrow_mut","borrow_mut","borrow_mut","borrow_mut","borrow_mut","borrow_mut","borrow_mut","borrow_mut","borrow_mut","borrow_mut","borrow_mut","clone","clone","clone_into","clone_into","compile_with_ptx_jit_args","config","copy_from","copy_to","deref","deref_mut","drop","drop","eq","equivalent","fmt","for_device","for_device","for_device","for_host","for_host","for_host","from","from","from","from","from","from","from","from","from","from","from","from","from","from","get_launch_package","grid","into","into","into","into","into","into","into","into","into","into","into","kernel","lend_to_cuda","lend_to_cuda_mut","move_to_cuda","new","new","new","new","on_compile","ptx_jit","shared_memory_size","split","stream","to_owned","to_owned","try_from","try_from","try_from","try_from","try_from","try_from","try_from","try_from","try_from","try_from","try_from","try_into","try_into","try_into","try_into","try_into","try_into","try_into","try_into","try_into","try_into","try_into","type_id","type_id","type_id","type_id","type_id","type_id","type_id","type_id","type_id","type_id","type_id","watcher","with_new","with_new","with_new","FitsIntoDeviceRegister","NoAliasing","SafeDeviceCopy","StackOnly","UnifiedHeapOnly","aliasing","alloc","device_copy","exchange","SplitSliceOverCudaThreadsConstStride","SplitSliceOverCudaThreadsDynamicStride","as_mut","as_mut","as_ref","as_ref","as_rust","as_rust","borrow","borrow","borrow","borrow","borrow","borrow","borrow_mut","borrow_mut","borrow_mut","borrow_mut","clone","clone","clone_into","clone_into","deref","deref","deref_mut","deref_mut","from","from","into","into","lend_to_cuda","lend_to_cuda","lend_to_cuda_mut","lend_to_cuda_mut","move_to_cuda","move_to_cuda","new","new","populate_graph","populate_graph","restore","restore","to_owned","to_owned","try_from","try_from","try_into","try_into","type_id","type_id","uninit","uninit","with_borrow_from_rust","with_borrow_from_rust","with_borrow_from_rust_mut","with_borrow_from_rust_mut","with_moved_from_rust","with_moved_from_rust","UnifiedAllocator","allocate","borrow","borrow_mut","deallocate","from","into","try_from","try_into","type_id","SafeDeviceCopyWrapper","as_rust","borrow","borrow","borrow_mut","clone","clone_into","fmt","from","from","from","from_mut","from_mut_slice","from_ref","from_slice","into","into_inner","into_mut","into_mut_slice","into_ref","into_slice","lend_to_cuda","lend_to_cuda_mut","move_to_cuda","populate_graph","restore","to_owned","try_from","try_into","type_id","uninit","with_borrow_from_rust","with_borrow_from_rust_mut","with_moved_from_rust","buffer","wrapper","CudaExchangeBuffer","CudaExchangeBufferDevice","CudaExchangeBufferHost","CudaExchangeItem","as_mut","borrow","borrow","borrow","borrow","borrow_mut","borrow_mut","borrow_mut","clone","clone_into","deref","deref","deref_mut","deref_mut","from","from","from","from_vec","into","into","into","lend_to_cuda","lend_to_cuda_mut","move_to_cuda","new","populate_graph","read","read","restore","to_owned","try_from","try_from","try_from","try_into","try_into","try_into","type_id","type_id","type_id","uninit","with_borrow_from_rust","with_borrow_from_rust_mut","with_moved_from_rust","write","write","ExchangeWrapperOnDevice","ExchangeWrapperOnHost","as_mut","as_ref","borrow","borrow","borrow_mut","borrow_mut","deref","deref_mut","from","from","into","into","move_to_device","move_to_host","new","try_from","try_from","try_into","try_into","type_id","type_id"],"q":[[0,"rust_cuda"],[10,"rust_cuda::common"],[63,"rust_cuda::device"],[79,"rust_cuda::device::utils"],[120,"rust_cuda::host"],[265,"rust_cuda::safety"],[270,"rust_cuda::utils"],[274,"rust_cuda::utils::aliasing"],[332,"rust_cuda::utils::alloc"],[342,"rust_cuda::utils::device_copy"],[376,"rust_cuda::utils::exchange"],[378,"rust_cuda::utils::exchange::buffer"],[427,"rust_cuda::utils::exchange::wrapper"]],"d":["Assertion in GPU kernel for one expression is true.","Assertion in GPU kernel for two expressions are equal.","Assertion in GPU kernel for two expressions are not equal.","","","","Alternative of <code>std::print!</code> using CUDA <code>vprintf</code> system-call","Alternative of <code>std::println!</code> using CUDA <code>vprintf</code> system-call","","","","Safety","","","","","","","Safety","","","","","Safety","Errors","","","","","","","","","","","","Returns the argument unchanged.","Returns the argument unchanged.","Returns the argument unchanged.","","","","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","","","","","Errors","","","","","","","","","","","","","","","","","","","","","Returns the argument unchanged.","Calls <code>U::from(self)</code>.","","","","","Safety","Safety","Safety","Dimension specified in kernel launching","Indices that the kernel code is running on","Memory allocator using CUDA malloc/free","","","","","","","","","","","","","","Returns the argument unchanged.","Returns the argument unchanged.","Returns the argument unchanged.","","","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","Errors","","Errors","Errors","","","","","","","","","","","","","","Returns the argument unchanged.","Returns the argument unchanged.","Returns the argument unchanged.","Returns the argument unchanged.","Returns the argument unchanged.","Returns the argument unchanged.","","","","Returns the argument unchanged.","Returns the argument unchanged.","Returns the argument unchanged.","Returns the argument unchanged.","Returns the argument unchanged.","","","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","","Lends an immutable copy of <code>&amp;self</code> to CUDA:","Lends a mutable copy of <code>&amp;mut self</code> to CUDA:","Moves <code>self</code> to CUDA iff <code>self</code> is <code>SafeDeviceCopy</code>","Errors","","Safety","Safety","Errors","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","Errors","Errors","Errors","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","Returns the argument unchanged.","Returns the argument unchanged.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","Returns the argument unchanged.","Calls <code>U::from(self)</code>.","","","","","","","","","","","","","Returns the argument unchanged.","","","","","","Calls <code>U::from(self)</code>.","","","","","","","","","","","","","","","","","","","","","When the <code>host</code> feature is <strong>not</strong> set, <code>CudaExchangeBuffer</code> …","When the <code>host</code> feature is set, <code>CudaExchangeBuffer</code> refers to …","When the <code>host</code> feature is <strong>not</strong> set, <code>CudaExchangeBuffer</code> …","","","","","","","","","","","","","","","","Returns the argument unchanged.","Returns the argument unchanged.","Returns the argument unchanged.","Errors","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","","","","Errors","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","Returns the argument unchanged.","Returns the argument unchanged.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","Errors","Errors","Errors","","","","","",""],"i":[0,0,0,0,0,0,0,0,0,0,52,0,52,0,0,0,0,12,0,0,2,3,2,12,52,4,3,2,4,3,2,3,3,4,4,4,4,3,2,66,66,66,4,3,2,0,4,3,2,52,3,4,3,2,4,3,2,4,3,2,4,3,2,0,0,23,23,23,23,23,23,23,23,23,23,0,67,67,67,0,0,0,25,28,0,0,25,29,28,25,29,28,25,29,28,25,29,28,0,0,25,29,28,29,0,25,29,28,25,29,28,25,29,28,29,28,29,28,29,28,36,0,68,0,0,0,0,0,0,0,0,68,0,0,0,0,0,36,0,31,31,32,33,43,36,34,69,17,38,37,31,32,40,33,43,36,34,69,17,38,37,31,32,40,33,32,33,32,33,34,43,37,37,38,38,38,37,33,33,33,31,32,40,31,32,40,43,36,34,69,17,38,38,38,37,37,31,32,40,33,68,33,43,36,34,69,17,38,37,31,32,40,33,43,70,70,70,34,17,31,32,68,33,33,17,43,32,33,43,36,34,69,17,38,37,31,32,40,33,43,36,34,69,17,38,37,31,32,40,33,43,36,34,69,17,38,37,31,32,40,33,43,31,32,40,0,0,0,0,0,0,0,0,0,0,0,49,50,49,50,49,50,49,49,49,50,50,50,49,49,50,50,49,50,49,50,49,50,49,50,49,50,49,50,49,50,49,50,49,50,49,50,49,50,49,50,49,50,49,50,49,50,49,50,49,50,49,50,49,50,49,50,0,57,57,57,57,57,57,57,57,57,0,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,0,0,0,0,0,0,60,62,61,61,60,62,61,60,60,60,62,61,62,61,62,61,60,61,62,61,60,61,61,61,61,60,60,60,61,60,62,61,60,62,61,60,62,61,60,60,61,61,61,60,60,0,0,64,64,65,64,65,64,65,65,65,64,65,64,65,64,65,65,64,65,64,65,64],"f":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,[[[2,[1]]],1],[[[3,[1]]],1],[[[2,[1]]],1],[4],[5,6],[[]],[[]],[[]],[[]],[[]],[[]],[[[3,[[0,[7,1]]]]],[[3,[[0,[7,1]]]]]],[[]],[[[4,[[0,[8,1,9]]]],10],11],[12,[[4,[12]]]],[[[0,[13,14]]],[[4,[[15,[[0,[13,14]]]]]]]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],0,[16],[16],[16],[[[17,[5]]],[[6,[5]]]],[[]],[[],18],[[],18],[[],18],[[],18],[[],18],[[],18],[[],19],[[],19],[[],19],[[],[[22,[[21,[[4,[[0,[20,8,1]]]]]]]]]],[[],[[22,[[21,[[3,[[0,[20,1]]]]]]]]]],[[],[[22,[[21,[[2,[[0,[20,1]]]]]]]]]],0,0,[[]],[[]],[23],[23],[[[23,[9]],10],11],[[]],[[]],[[],18],[[],18],[[],19],0,[[[3,[4]],24]],[[[2,[4]],24]],[[[2,[4]],24]],0,0,0,[[25,26],27],[[28,29],30],[[],29],[[],28],[[]],[[]],[[]],[[]],[[]],[[]],[[25,27,26]],[[29,10],11],[[28,10],11],[[]],[[]],[[]],[[],29],[[],30],[[]],[[]],[[]],[29,30],[[],28],[[],18],[[],18],[[],18],[[],18],[[],18],[[],18],[[],19],[[],19],[[],19],0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,[[[31,[1]]],[[31,[1]]]],[[[31,[1]]],[[32,[1]]]],[[[32,[1]]],[[32,[1]]]],0,[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[[32,[1]]],[[32,[1]]]],[33,33],[[]],[[]],[[[34,[8]],35],[[6,[36]]]],0,[[[37,[1]],1],6],[[[37,[1]],1],6],[[[38,[0]]]],[[[38,[0]]]],[[[38,[0]]]],[[[37,[1]]]],[[33,33],39],[[],39],[[33,10],11],[[[31,[1]]],[[2,[1]]]],[[[32,[1]]],[[3,[1]]]],[[[40,[[0,[13,1]]]]],[[2,[[0,[13,1]]]]]],[[[31,[1]]],1],[[[32,[1]]],1],[[[40,[[0,[13,1]]]]],[[0,[13,1]]]],[[]],[[]],[[]],[[]],[[]],[[]],[41],0,[[[42,[1]]],[[37,[1]]]],[[]],[[]],[[]],[[]],[[]],[[],43],0,[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],0,[24,[[18,[[45,[44]]]]]],[24,[[18,[[45,[44]]]]]],[[[0,[8,13]],24],[[18,[[45,[44]]]]]],[[46,46],[[6,[[34,[8]]]]]],[[5,5],[[17,[5,5]]]],[[[37,[1]],1],[[31,[1]]]],[[[37,[1]],1],[[32,[1]]]],[47,6],0,0,[[[17,[5,5]]]],0,[[]],[[]],[[],18],[[],18],[[],18],[[],18],[[],18],[[],18],[[],18],[[],18],[[],18],[[],18],[[],18],[[],18],[[],18],[[],18],[[],18],[[],18],[[],18],[[],18],[[],18],[[],18],[[],18],[[],18],[[],19],[[],19],[[],19],[[],19],[[],19],[[],19],[[],19],[[],19],[[],19],[[],19],[[],19],0,[[1,24],[[18,[[45,[44]]]]]],[[1,24],[[18,[[45,[44]]]]]],[[[0,[13,1]],24],[[18,[[45,[44]]]]]],0,0,0,0,0,0,0,0,0,0,0,[[[49,[48]]]],[[[50,[48]]]],[[[49,[51]]]],[[[50,[51]]]],[[[4,[[49,[[4,[12]]]]]]]],[[[4,[[50,[[4,[12]]]]]]]],[[]],[[[49,[52]],5],6],[[[49,[53]]]],[[]],[[[50,[53]]]],[[[50,[52]],5],6],[[]],[[[49,[54]]]],[[]],[[[50,[54]]]],[[[49,[7]]],[[49,[7]]]],[[[50,[7]]],[[50,[7]]]],[[]],[[]],[[[49,[55]]]],[[[50,[55]]]],[[[49,[56]]]],[[[50,[56]]]],[[]],[[]],[[]],[[]],[24,[[18,[[45,[44]]]]]],[24,[[18,[[45,[44]]]]]],[24,[[18,[[45,[44]]]]]],[24,[[18,[[45,[44]]]]]],[24,[[18,[[45,[44]]]]]],[24,[[18,[[45,[44]]]]]],[[],49],[30,50],[16],[16],[[[49,[52]],[17,[5]]],[[6,[5]]]],[[[50,[52]],[17,[5]]],[[6,[5]]]],[[]],[[]],[[],18],[[],18],[[],18],[[],18],[[],19],[[],19],[[],[[22,[[21,[[49,[20]]]]]]]],[[],[[22,[[21,[[50,[20]]]]]]]],[[[3,[4]],24]],[[[3,[4]],24]],[[[2,[4]],24]],[[[2,[4]],24]],[[[2,[4]],24]],[[[2,[4]],24]],0,[[57,26],[[18,[58,59]]]],[[]],[[]],[[57,[58,[27]],26]],[[]],[[]],[[],18],[[],18],[[],19],0,[[[4,[[15,[[0,[13,14]]]]]]]],[[[15,[[0,[13,14]]]],5],6],[[]],[[]],[[[15,[[0,[13,14,7]]]]],[[15,[[0,[13,14,7]]]]]],[[]],[[[15,[[0,[13,14,9]]]],10],11],[41],[[]],[[[0,[13,14]]],[[15,[[0,[13,14]]]]]],[[[0,[13,14]]],[[15,[[0,[13,14]]]]]],[[]],[[[0,[13,14]]],[[15,[[0,[13,14]]]]]],[[]],[[]],[[[15,[[0,[13,14]]]]],[[0,[13,14]]]],[[[15,[[0,[13,14]]]]],[[0,[13,14]]]],[[]],[[[15,[[0,[13,14]]]]],[[0,[13,14]]]],[[]],[24,[[18,[[45,[44]]]]]],[24,[[18,[[45,[44]]]]]],[24,[[18,[[45,[44]]]]]],[16],[[[15,[[0,[13,14]]]],[17,[5]]],[[6,[5]]]],[[]],[[],18],[[],18],[[],19],[[],[[22,[[21,[[15,[[0,[13,14,20]]]]]]]]]],[[[3,[4]],24]],[[[2,[4]],24]],[[[2,[4]],24]],0,0,0,0,0,0,[[[60,[13]]],13],[[]],[[]],[[[61,[[0,[13,14]]]],5],6],[[]],[[]],[[]],[[]],[[[60,[[0,[7,13]]]]],[[60,[[0,[7,13]]]]]],[[]],[[[62,[13]]]],[[[61,[[0,[13,14]]]]]],[[[62,[13]]]],[[[61,[[0,[13,14]]]]]],[[]],[[]],[[]],[[[63,[[0,[13,14]]]]],[[6,[[61,[[0,[13,14]]]]]]]],[[]],[[]],[[]],[24,[[18,[[45,[44]]]]]],[24,[[18,[[45,[44]]]]]],[24,[[18,[[45,[44]]]]]],[[[0,[7,13,14]],30],[[6,[[61,[[0,[7,13,14]]]]]]]],[16],[[[60,[13]]],13],[[[60,[13]]],13],[[[61,[[0,[13,14]]]],[17,[5]]],[[6,[5]]]],[[]],[[],18],[[],18],[[],18],[[],18],[[],18],[[],18],[[],19],[[],19],[[],19],[[],[[22,[[21,[[60,[[0,[20,13]]]]]]]]]],[[[3,[4]],24]],[[[2,[4]],24]],[[[2,[4]],24]],[[[60,[13]],13]],[[[60,[13]],13]],0,0,[[[64,[52]]],[[31,[4]]]],[[[64,[52]]],[[32,[4]]]],[[]],[[]],[[]],[[]],[[[65,[52]]]],[[[65,[52]]]],[[]],[[]],[[]],[[]],[[[65,[52]]],[[6,[[64,[52]]]]]],[[[64,[52]]],[[6,[[65,[52]]]]]],[52,[[6,[[65,[52]]]]]],[[],18],[[],18],[[],18],[[],18],[[],19],[[],19]],"c":[],"p":[[8,"DeviceCopy"],[3,"DeviceMutRef"],[3,"DeviceConstRef"],[3,"DeviceAccessible"],[8,"CudaAlloc"],[6,"CudaResult"],[8,"Clone"],[8,"Sized"],[8,"Debug"],[3,"Formatter"],[6,"Result"],[8,"CudaAsRust"],[8,"SafeDeviceCopy"],[8,"TypeGraphLayout"],[3,"SafeDeviceCopyWrapper"],[3,"TypeLayoutGraph"],[3,"CombinedCudaAlloc"],[4,"Result"],[3,"TypeId"],[8,"TypeLayout"],[19,"MaybeUninit"],[4,"MaybeUninhabited"],[3,"ShallowCopy"],[8,"FnOnce"],[3,"PTXAllocator"],[3,"Layout"],[15,"u8"],[3,"Idx3"],[3,"Dim3"],[15,"usize"],[3,"HostAndDeviceMutRef"],[3,"HostAndDeviceConstRef"],[3,"LaunchConfig"],[3,"TypedKernel"],[4,"Option"],[4,"KernelJITResult"],[3,"HostDeviceBox"],[3,"CudaDropWrapper"],[15,"bool"],[3,"HostAndDeviceOwned"],[15,"never"],[3,"DeviceBox"],[3,"LaunchPackage"],[4,"CudaError"],[8,"From"],[15,"str"],[3,"Function"],[8,"AsMut"],[3,"SplitSliceOverCudaThreadsConstStride"],[3,"SplitSliceOverCudaThreadsDynamicStride"],[8,"AsRef"],[8,"RustToCuda"],[8,"Borrow"],[8,"BorrowMut"],[8,"Deref"],[8,"DerefMut"],[3,"UnifiedAllocator"],[3,"NonNull"],[3,"AllocError"],[3,"CudaExchangeItem"],[3,"CudaExchangeBuffer"],[3,"CudaExchangeBufferDevice"],[3,"Vec"],[3,"ExchangeWrapperOnDevice"],[3,"ExchangeWrapperOnHost"],[8,"RustToCudaProxy"],[8,"BorrowFromRust"],[8,"Launcher"],[3,"NullCudaAlloc"],[8,"LendToCuda"]]},\
"rust_cuda_derive":{"doc":"","t":"YX","n":["LendRustToCuda","kernel"],"q":[[0,"rust_cuda_derive"]],"d":["",""],"i":[0,0],"f":[0,0],"c":[],"p":[]},\
"rust_cuda_ptx_jit":{"doc":"","t":"NDDENLLLLLLLLLLLLLLLLLLLLLLLLLL","n":["Cached","CudaKernel","PtxJITCompiler","PtxJITResult","Recomputed","borrow","borrow","borrow","borrow_mut","borrow_mut","borrow_mut","drop","from","from","from","get_function","into","into","into","new","new","try_from","try_from","try_from","try_into","try_into","try_into","type_id","type_id","type_id","with_arguments"],"q":[[0,"rust_cuda_ptx_jit"]],"d":["","","","","","","","","","","","","Returns the argument unchanged.","Returns the argument unchanged.","Returns the argument unchanged.","","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","","Errors","","","","","","","","","",""],"i":[9,0,0,0,9,4,9,1,4,9,1,1,4,9,1,1,4,9,1,4,1,4,9,1,4,9,1,4,9,1,4],"f":[0,0,0,0,0,[[]],[[]],[[]],[[]],[[]],[[]],[1],[[]],[[]],[[]],[1,2],[[]],[[]],[[]],[3,4],[[3,3],[[5,[1]]]],[[],6],[[],6],[[],6],[[],6],[[],6],[[],6],[[],7],[[],7],[[],7],[[4,8],9]],"c":[],"p":[[3,"CudaKernel"],[3,"Function"],[3,"CStr"],[3,"PtxJITCompiler"],[6,"CudaResult"],[4,"Result"],[3,"TypeId"],[4,"Option"],[4,"PtxJITResult"]]}\
}');
if (typeof window !== 'undefined' && window.initSearch) {window.initSearch(searchIndex)};
if (typeof exports !== 'undefined') {exports.searchIndex = searchIndex};
