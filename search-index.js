var searchIndex = JSON.parse('{\
"rust_cuda":{"doc":"","t":"OOOAAAOOBBBBAAQIQDDDYQIILLLKKLLLLLLLLLLLLLLKKKLLLXKLLLLLLLLLLIDLLLLLLLLLLAKKKDDDLLFFLLLLLLLLLLLLFFLLLLFLLLLLLLLLMMMMMMNDQIDIDDDDEQDDIIDNDLLLMLLLLLLLLLLLLLLLLLLLLLLLLLLLMLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLKMLLLLLLLLLLLMKKKLLLLLMMLMLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLMLLLIIIIIAAAADDLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLDLLLLLLLLLDLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLAADDDDLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLDDLLLLLLLLLLLLLLLLLLLLL","n":["assert","assert_eq","assert_ne","common","device","host","print","println","ptx_jit","rustacuda","rustacuda_core","rustacuda_derive","safety","utils","CudaAllocation","CudaAsRust","CudaRepresentation","DeviceAccessible","DeviceConstRef","DeviceMutRef","LendRustToCuda","RustRepresentation","RustToCuda","RustToCudaProxy","as_mut","as_ref","as_ref","as_rust","borrow","borrow","borrow","borrow","borrow_mut","borrow_mut","borrow_mut","clone","clone_into","fmt","from","from","from","from","from","from_mut","from_ref","into","into","into","into","kernel","restore","to_owned","try_from","try_from","try_from","try_into","try_into","try_into","type_id","type_id","type_id","BorrowFromRust","ShallowCopy","borrow","borrow_mut","deref","deref_mut","fmt","from","into","try_from","try_into","type_id","utils","with_borrow_from_rust","with_borrow_from_rust_mut","with_moved_from_rust","Dim3","Idx3","PTXAllocator","alloc","as_id","block_dim","block_idx","borrow","borrow","borrow","borrow_mut","borrow_mut","borrow_mut","dealloc","fmt","fmt","from","from","from","grid_dim","index","into","into","into","size","thread_idx","try_from","try_from","try_from","try_into","try_into","try_into","type_id","type_id","type_id","x","x","y","y","z","z","Cached","CombinedCudaAlloc","CompilationWatcher","CudaAlloc","CudaDropWrapper","EmptyCudaAlloc","HostAndDeviceConstRef","HostAndDeviceMutRef","HostAndDeviceOwned","HostDeviceBox","KernelJITResult","KernelTraitObject","LaunchConfig","LaunchPackage","Launcher","LendToCuda","NullCudaAlloc","Recompiled","TypedKernel","as_mut","as_ref","as_ref","block","borrow","borrow","borrow","borrow","borrow","borrow","borrow","borrow","borrow","borrow","borrow","borrow_mut","borrow_mut","borrow_mut","borrow_mut","borrow_mut","borrow_mut","borrow_mut","borrow_mut","borrow_mut","borrow_mut","borrow_mut","clone","clone","clone_into","clone_into","compile_with_ptx_jit_args","config","copy_from","copy_to","deref","deref_mut","drop","drop","eq","equivalent","equivalent","fmt","for_device","for_device","for_device","for_host","for_host","for_host","from","from","from","from","from","from","from","from","from","from","from","from","from","from","get_launch_package","grid","into","into","into","into","into","into","into","into","into","into","into","kernel","lend_to_cuda","lend_to_cuda_mut","move_to_cuda","new","new","new","new","on_compile","ptx_jit","shared_memory_size","split","stream","to_owned","to_owned","try_from","try_from","try_from","try_from","try_from","try_from","try_from","try_from","try_from","try_from","try_from","try_into","try_into","try_into","try_into","try_into","try_into","try_into","try_into","try_into","try_into","try_into","type_id","type_id","type_id","type_id","type_id","type_id","type_id","type_id","type_id","type_id","type_id","watcher","with_new","with_new","with_new","FitsIntoDeviceRegister","NoAliasing","SafeDeviceCopy","StackOnly","UnifiedHeapOnly","aliasing","alloc","device_copy","exchange","SplitSliceOverCudaThreadsConstStride","SplitSliceOverCudaThreadsDynamicStride","as_mut","as_mut","as_ref","as_ref","as_rust","as_rust","borrow","borrow","borrow","borrow","borrow","borrow","borrow_mut","borrow_mut","borrow_mut","borrow_mut","clone","clone","clone_into","clone_into","deref","deref","deref_mut","deref_mut","from","from","into","into","lend_to_cuda","lend_to_cuda","lend_to_cuda_mut","lend_to_cuda_mut","move_to_cuda","move_to_cuda","new","new","restore","restore","to_owned","to_owned","try_from","try_from","try_into","try_into","type_id","type_id","with_borrow_from_rust","with_borrow_from_rust","with_borrow_from_rust_mut","with_borrow_from_rust_mut","with_moved_from_rust","with_moved_from_rust","UnifiedAllocator","allocate","borrow","borrow_mut","deallocate","from","into","try_from","try_into","type_id","SafeDeviceCopyWrapper","as_rust","borrow","borrow","borrow_mut","clone","clone_into","fmt","from","from","from","from_mut","from_mut_slice","from_ref","from_slice","into","into_inner","into_mut","into_mut_slice","into_ref","into_slice","lend_to_cuda","lend_to_cuda_mut","move_to_cuda","restore","to_owned","try_from","try_into","type_id","with_borrow_from_rust","with_borrow_from_rust_mut","with_moved_from_rust","buffer","wrapper","CudaExchangeBuffer","CudaExchangeBufferDevice","CudaExchangeBufferHost","CudaExchangeItem","as_mut","borrow","borrow","borrow","borrow","borrow_mut","borrow_mut","borrow_mut","clone","clone_into","deref","deref","deref_mut","deref_mut","from","from","from","from_vec","into","into","into","lend_to_cuda","lend_to_cuda_mut","move_to_cuda","new","read","read","restore","to_owned","try_from","try_from","try_from","try_into","try_into","try_into","type_id","type_id","type_id","with_borrow_from_rust","with_borrow_from_rust_mut","with_moved_from_rust","write","write","ExchangeWrapperOnDevice","ExchangeWrapperOnHost","as_mut","as_ref","borrow","borrow","borrow_mut","borrow_mut","deref","deref_mut","from","from","into","into","move_to_device","move_to_host","new","try_from","try_from","try_into","try_into","type_id","type_id"],"q":[[0,"rust_cuda"],[14,"rust_cuda::common"],[61,"rust_cuda::device"],[77,"rust_cuda::device::utils"],[118,"rust_cuda::host"],[264,"rust_cuda::safety"],[269,"rust_cuda::utils"],[273,"rust_cuda::utils::aliasing"],[327,"rust_cuda::utils::alloc"],[337,"rust_cuda::utils::device_copy"],[369,"rust_cuda::utils::exchange"],[371,"rust_cuda::utils::exchange::buffer"],[418,"rust_cuda::utils::exchange::wrapper"],[441,"rustacuda_core::memory"],[442,"rustacuda::error"],[443,"core::clone"],[444,"core::fmt"],[445,"core::fmt"],[446,"core::fmt"],[447,"core::result"],[448,"core::any"],[449,"core::ops::function"],[450,"core::alloc::layout"],[451,"core::option"],[452,"rustacuda::memory::device::device_box"],[453,"rustacuda::error"],[454,"rustacuda::function"],[455,"core::convert"],[456,"core::borrow"],[457,"core::ops::deref"],[458,"core::alloc"],[459,"alloc::vec"]],"d":["Assertion in GPU kernel for one expression is true.","Assertion in GPU kernel for two expressions are equal.","Assertion in GPU kernel for two expressions are not equal.","","","","Alternative of <code>std::print!</code> using CUDA <code>vprintf</code> system-call","Alternative of <code>std::println!</code> using CUDA <code>vprintf</code> system-call","","","","","","","","Safety","","","","","","","Safety","","","","","Safety","Errors","","","","","","","","","","","Returns the argument unchanged.","","Returns the argument unchanged.","Returns the argument unchanged.","","","","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","","Errors","","","","","","","","","","","","","","","","","","Returns the argument unchanged.","Calls <code>U::from(self)</code>.","","","","","Safety","Safety","Safety","Dimension specified in kernel launching","Indices that the kernel code is running on","Memory allocator using CUDA malloc/free","","","","","","","","","","","","","","Returns the argument unchanged.","Returns the argument unchanged.","Returns the argument unchanged.","","","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","Errors","","Errors","Errors","","","","","","","","","","","","","","","Returns the argument unchanged.","Returns the argument unchanged.","Returns the argument unchanged.","Returns the argument unchanged.","Returns the argument unchanged.","","","Returns the argument unchanged.","","Returns the argument unchanged.","Returns the argument unchanged.","Returns the argument unchanged.","Returns the argument unchanged.","Returns the argument unchanged.","","","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","","Lends an immutable copy of <code>&amp;self</code> to CUDA:","Lends a mutable copy of <code>&amp;mut self</code> to CUDA:","Moves <code>self</code> to CUDA iff <code>self</code> is <code>SafeDeviceCopy</code>","Errors","","Safety","Safety","Errors","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","Errors","Errors","Errors","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","Returns the argument unchanged.","Returns the argument unchanged.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","Returns the argument unchanged.","Calls <code>U::from(self)</code>.","","","","","","","","","","","","","Returns the argument unchanged.","","","","","","Calls <code>U::from(self)</code>.","","","","","","","","","","","","","","","","","","","When the <code>host</code> feature is <strong>not</strong> set, <code>CudaExchangeBuffer</code> …","When the <code>host</code> feature is set, <code>CudaExchangeBuffer</code> refers to …","When the <code>host</code> feature is <strong>not</strong> set, <code>CudaExchangeBuffer</code> …","","","","","","","","","","","","","","","","Returns the argument unchanged.","Returns the argument unchanged.","Returns the argument unchanged.","Errors","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","","","","Errors","","","","","","","","","","","","","","","","","","","","","","","","","","","","","Returns the argument unchanged.","Returns the argument unchanged.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","Errors","Errors","Errors","","","","","",""],"i":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,50,0,50,0,0,0,0,17,0,0,1,3,1,17,50,4,3,1,4,3,1,3,3,4,4,4,4,3,1,64,64,64,4,3,1,0,50,3,4,3,1,4,3,1,4,3,1,0,0,20,20,20,20,20,20,20,20,20,20,0,65,65,65,0,0,0,22,25,0,0,22,26,25,22,26,25,22,26,25,22,26,25,0,0,22,26,25,26,0,22,26,25,22,26,25,22,26,25,26,25,26,25,26,25,34,0,66,0,0,0,0,0,0,0,0,66,0,0,0,0,0,34,0,28,28,29,30,41,34,31,67,5,36,35,28,29,38,30,41,34,31,67,5,36,35,28,29,38,30,29,30,29,30,31,41,35,35,36,36,36,35,30,30,30,30,28,29,38,28,29,38,41,34,31,67,5,36,36,36,35,35,28,29,38,30,66,30,41,34,31,67,5,36,35,28,29,38,30,41,68,68,68,31,5,28,29,66,30,30,5,41,29,30,41,34,31,67,5,36,35,28,29,38,30,41,34,31,67,5,36,35,28,29,38,30,41,34,31,67,5,36,35,28,29,38,30,41,28,29,38,0,0,0,0,0,0,0,0,0,0,0,46,48,46,48,46,48,46,46,46,48,48,48,46,46,48,48,46,48,46,48,46,48,46,48,46,48,46,48,46,48,46,48,46,48,46,48,46,48,46,48,46,48,46,48,46,48,46,48,46,48,46,48,0,55,55,55,55,55,55,55,55,55,0,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,0,0,0,0,0,0,58,60,59,59,58,60,59,58,58,58,60,59,60,59,60,59,58,59,60,59,58,59,59,59,59,58,58,59,58,60,59,58,60,59,58,60,59,58,59,59,59,58,58,0,0,62,62,63,62,63,62,63,63,63,62,63,62,63,62,63,63,62,63,62,63,62],"f":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,[[[1,[-1]]],-1,2],[[[3,[-1]]],-1,2],[[[1,[-1]]],-1,2],[[[4,[-1]]],[],[]],[[-1,-2],[[7,[[6,[4,[5,[-2]]]]]]],[],8],[-1,-2,[],[]],[-1,-2,[],[]],[-1,-2,[],[]],[-1,-2,[],[]],[-1,-2,[],[]],[-1,-2,[],[]],[[[3,[-1]]],[[3,[-1]]],[9,2]],[[-1,-2],6,[],[]],[[[4,[-1]],10],11,[12,2,13]],[-1,[[4,[[14,[-1]]]]],[15,16]],[-1,-1,[]],[-1,[[4,[-1]]],17],[-1,-1,[]],[-1,-1,[]],[-1,-2,[],[]],[-1,-2,[],[]],[-1,-2,[],[]],[-1,-2,[],[]],[-1,-2,[],[]],[-1,-2,[],[]],0,[[-1,[5,[-2]]],[[7,[-2]]],[],8],[-1,-2,[],[]],[-1,[[18,[-2]]],[],[]],[-1,[[18,[-2]]],[],[]],[-1,[[18,[-2]]],[],[]],[-1,[[18,[-2]]],[],[]],[-1,[[18,[-2]]],[],[]],[-1,[[18,[-2]]],[],[]],[-1,19,[]],[-1,19,[]],[-1,19,[]],0,0,[-1,-2,[],[]],[-1,-2,[],[]],[[[20,[-1]]],[],[]],[[[20,[-1]]],[],[]],[[[20,[-1]],10],11,13],[-1,-1,[]],[-1,-2,[],[]],[-1,[[18,[-2]]],[],[]],[-1,[[18,[-2]]],[],[]],[-1,19,[]],0,[[[3,[4]],-1],-2,21,[]],[[[1,[4]],-1],-2,21,[]],[[[1,[4]],-1],-2,21,[]],0,0,0,[[22,23],24],[[25,26],27],[[],26],[[],25],[-1,-2,[],[]],[-1,-2,[],[]],[-1,-2,[],[]],[-1,-2,[],[]],[-1,-2,[],[]],[-1,-2,[],[]],[[22,24,23],6],[[26,10],11],[[25,10],11],[-1,-1,[]],[-1,-1,[]],[-1,-1,[]],[[],26],[[],27],[-1,-2,[],[]],[-1,-2,[],[]],[-1,-2,[],[]],[26,27],[[],25],[-1,[[18,[-2]]],[],[]],[-1,[[18,[-2]]],[],[]],[-1,[[18,[-2]]],[],[]],[-1,[[18,[-2]]],[],[]],[-1,[[18,[-2]]],[],[]],[-1,[[18,[-2]]],[],[]],[-1,19,[]],[-1,19,[]],[-1,19,[]],0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,[[[28,[-1]]],[[28,[-1]]],2],[[[28,[-1]]],[[29,[-1]]],2],[[[29,[-1]]],[[29,[-1]]],2],0,[-1,-2,[],[]],[-1,-2,[],[]],[-1,-2,[],[]],[-1,-2,[],[]],[-1,-2,[],[]],[-1,-2,[],[]],[-1,-2,[],[]],[-1,-2,[],[]],[-1,-2,[],[]],[-1,-2,[],[]],[-1,-2,[],[]],[-1,-2,[],[]],[-1,-2,[],[]],[-1,-2,[],[]],[-1,-2,[],[]],[-1,-2,[],[]],[-1,-2,[],[]],[-1,-2,[],[]],[-1,-2,[],[]],[-1,-2,[],[]],[-1,-2,[],[]],[-1,-2,[],[]],[[[29,[-1]]],[[29,[-1]]],2],[30,30],[[-1,-2],6,[],[]],[[-1,-2],6,[],[]],[[[31,[-1]],[33,[[32,[[33,[[32,[24]]]]]]]]],[[7,[34]]],12],0,[[[35,[-1]],-1],[[7,[6]]],2],[[[35,[-1]],-1],[[7,[6]]],2],[[[36,[-1]]],[],0],[[[36,[-1]]],[],0],[[[36,[-1]]],6,0],[[[35,[-1]]],6,2],[[30,30],37],[[-1,-2],37,[],[]],[[-1,-2],37,[],[]],[[30,10],11],[[[28,[-1]]],[[1,[-1]]],2],[[[29,[-1]]],[[3,[-1]]],2],[[[38,[-1]]],[[1,[-1]]],[15,2]],[[[28,[-1]]],-1,2],[[[29,[-1]]],-1,2],[[[38,[-1]]],-1,[15,2]],[-1,-1,[]],[-1,-1,[]],[-1,-1,[]],[-1,-1,[]],[-1,-1,[]],[-1,[[36,[-1]]],0],[39,-1,[]],[-1,-1,[]],[[[40,[-1]]],[[35,[-1]]],2],[-1,-1,[]],[-1,-1,[]],[-1,-1,[]],[-1,-1,[]],[-1,-1,[]],[-1,[[41,[-1]]],[]],0,[-1,-2,[],[]],[-1,-2,[],[]],[-1,-2,[],[]],[-1,-2,[],[]],[-1,-2,[],[]],[-1,-2,[],[]],[-1,-2,[],[]],[-1,-2,[],[]],[-1,-2,[],[]],[-1,-2,[],[]],[-1,-2,[],[]],0,[[-1,-2],[[18,[-3,-4]]],[],21,[],[[43,[42]]]],[[-1,-2],[[18,[-3,-4]]],[],21,[],[[43,[42]]]],[[-1,-2],[[18,[-3,-4]]],[12,15],21,[],[[43,[42]]]],[[44,44],[[7,[[31,[-1]]]]],12],[[-1,-2],[[5,[-1,-2]]],8,8],[[[35,[-1]],-1],[[28,[-1]]],2],[[[35,[-1]],-1],[[29,[-1]]],2],[45,[[7,[6]]]],0,0,[[[5,[-1,-2]]],[[6,[-1,-2]]],8,8],0,[-1,-2,[],[]],[-1,-2,[],[]],[-1,[[18,[-2]]],[],[]],[-1,[[18,[-2]]],[],[]],[-1,[[18,[-2]]],[],[]],[-1,[[18,[-2]]],[],[]],[-1,[[18,[-2]]],[],[]],[-1,[[18,[-2]]],[],[]],[-1,[[18,[-2]]],[],[]],[-1,[[18,[-2]]],[],[]],[-1,[[18,[-2]]],[],[]],[-1,[[18,[-2]]],[],[]],[-1,[[18,[-2]]],[],[]],[-1,[[18,[-2]]],[],[]],[-1,[[18,[-2]]],[],[]],[-1,[[18,[-2]]],[],[]],[-1,[[18,[-2]]],[],[]],[-1,[[18,[-2]]],[],[]],[-1,[[18,[-2]]],[],[]],[-1,[[18,[-2]]],[],[]],[-1,[[18,[-2]]],[],[]],[-1,[[18,[-2]]],[],[]],[-1,[[18,[-2]]],[],[]],[-1,[[18,[-2]]],[],[]],[-1,19,[]],[-1,19,[]],[-1,19,[]],[-1,19,[]],[-1,19,[]],[-1,19,[]],[-1,19,[]],[-1,19,[]],[-1,19,[]],[-1,19,[]],[-1,19,[]],0,[[-1,-2],[[18,[-3,-4]]],2,21,[],[[43,[42]]]],[[-1,-2],[[18,[-3,-4]]],2,21,[],[[43,[42]]]],[[-1,-2],[[18,[-3,-4]]],[15,2],21,[],[[43,[42]]]],0,0,0,0,0,0,0,0,0,0,0,[[[46,[-2]]],[[32,[-1]]],[],[[47,[[32,[-1]]]]]],[[[48,[-2]]],[[32,[-1]]],[],[[47,[[32,[-1]]]]]],[[[46,[-2]]],[[32,[-1]]],[],[[49,[[32,[-1]]]]]],[[[48,[-2]]],[[32,[-1]]],[],[[49,[[32,[-1]]]]]],[[[4,[[46,[[4,[-1]]]]]]],[],17],[[[4,[[48,[[4,[-1]]]]]]],[],17],[[[46,[-1]],-2],[[7,[[6,[4,[5,[-2]]]]]]],50,8],[[[46,[-2]]],[[32,[-1]]],[],[[51,[[32,[-1]]]]]],[-1,-2,[],[]],[[[48,[-2]]],[[32,[-1]]],[],[[51,[[32,[-1]]]]]],[-1,-2,[],[]],[[[48,[-1]],-2],[[7,[[6,[4,[5,[-2]]]]]]],50,8],[-1,-2,[],[]],[[[46,[-2]]],[[32,[-1]]],[],[[52,[[32,[-1]]]]]],[[[48,[-2]]],[[32,[-1]]],[],[[52,[[32,[-1]]]]]],[-1,-2,[],[]],[[[46,[-1]]],[[46,[-1]]],9],[[[48,[-1]]],[[48,[-1]]],9],[[-1,-2],6,[],[]],[[-1,-2],6,[],[]],[[[46,[-1]]],[],53],[[[48,[-1]]],[],53],[[[46,[-1]]],[],54],[[[48,[-1]]],[],54],[-1,-1,[]],[-1,-1,[]],[-1,-2,[],[]],[-1,-2,[],[]],[[-1,-2],[[18,[-3,-4]]],[],21,[],[[43,[42]]]],[[-1,-2],[[18,[-3,-4]]],[],21,[],[[43,[42]]]],[[-1,-2],[[18,[-3,-4]]],[],21,[],[[43,[42]]]],[[-1,-2],[[18,[-3,-4]]],[],21,[],[[43,[42]]]],[[-1,-2],[[18,[-3,-4]]],[],21,[],[[43,[42]]]],[[-1,-2],[[18,[-3,-4]]],[],21,[],[[43,[42]]]],[-1,[[46,[-1]]],[]],[[-1,27],[[48,[-1]]],[]],[[[46,[-1]],[5,[-2]]],[[7,[-2]]],50,8],[[[48,[-1]],[5,[-2]]],[[7,[-2]]],50,8],[-1,-2,[],[]],[-1,-2,[],[]],[-1,[[18,[-2]]],[],[]],[-1,[[18,[-2]]],[],[]],[-1,[[18,[-2]]],[],[]],[-1,[[18,[-2]]],[],[]],[-1,19,[]],[-1,19,[]],[[[3,[4]],-1],-2,21,[]],[[[3,[4]],-1],-2,21,[]],[[[1,[4]],-1],-2,21,[]],[[[1,[4]],-1],-2,21,[]],[[[1,[4]],-1],-2,21,[]],[[[1,[4]],-1],-2,21,[]],0,[[55,23],[[18,[[56,[[32,[24]]]],57]]]],[-1,-2,[],[]],[-1,-2,[],[]],[[55,[56,[24]],23],6],[-1,-1,[]],[-1,-2,[],[]],[-1,[[18,[-2]]],[],[]],[-1,[[18,[-2]]],[],[]],[-1,19,[]],0,[[[4,[[14,[-1]]]]],[],[15,16]],[[[14,[-1]],-2],[[7,[[6,[4,[5,[-2]]]]]]],[15,16],8],[-1,-2,[],[]],[-1,-2,[],[]],[[[14,[-1]]],[[14,[-1]]],[15,16,9]],[[-1,-2],6,[],[]],[[[14,[-1]],10],11,[15,16,13]],[39,-1,[]],[-1,-1,[]],[-1,[[14,[-1]]],[15,16]],[-1,[[14,[-1]]],[15,16]],[[[32,[-1]]],[[32,[[14,[-1]]]]],[15,16]],[-1,[[14,[-1]]],[15,16]],[[[32,[-1]]],[[32,[[14,[-1]]]]],[15,16]],[-1,-2,[],[]],[[[14,[-1]]],-1,[15,16]],[[[14,[-1]]],-1,[15,16]],[[[32,[[14,[-1]]]]],[[32,[-1]]],[15,16]],[[[14,[-1]]],-1,[15,16]],[[[32,[[14,[-1]]]]],[[32,[-1]]],[15,16]],[[-1,-2],[[18,[-3,-4]]],[],21,[],[[43,[42]]]],[[-1,-2],[[18,[-3,-4]]],[],21,[],[[43,[42]]]],[[-1,-2],[[18,[-3,-4]]],[],21,[],[[43,[42]]]],[[[14,[-1]],[5,[-2]]],[[7,[-2]]],[15,16],8],[-1,-2,[],[]],[-1,[[18,[-2]]],[],[]],[-1,[[18,[-2]]],[],[]],[-1,19,[]],[[[3,[4]],-1],-2,21,[]],[[[1,[4]],-1],-2,21,[]],[[[1,[4]],-1],-2,21,[]],0,0,0,0,0,0,[[[58,[-1]]],-1,15],[-1,-2,[],[]],[[[59,[-1]],-2],[[7,[[6,[4,[5,[-2]]]]]]],[15,16],8],[-1,-2,[],[]],[-1,-2,[],[]],[-1,-2,[],[]],[-1,-2,[],[]],[-1,-2,[],[]],[[[58,[-1]]],[[58,[-1]]],[9,15]],[[-1,-2],6,[],[]],[[[60,[-1]]],[],15],[[[59,[-1]]],[],[15,16]],[[[60,[-1]]],[],15],[[[59,[-1]]],[],[15,16]],[-1,-1,[]],[-1,-1,[]],[-1,-1,[]],[[[61,[-1]]],[[7,[[59,[-1]]]]],[15,16]],[-1,-2,[],[]],[-1,-2,[],[]],[-1,-2,[],[]],[[-1,-2],[[18,[-3,-4]]],[],21,[],[[43,[42]]]],[[-1,-2],[[18,[-3,-4]]],[],21,[],[[43,[42]]]],[[-1,-2],[[18,[-3,-4]]],[],21,[],[[43,[42]]]],[[-1,27],[[7,[[59,[-1]]]]],[9,15,16]],[[[58,[-1]]],-1,15],[[[58,[-1]]],-1,15],[[[59,[-1]],[5,[-2]]],[[7,[-2]]],[15,16],8],[-1,-2,[],[]],[-1,[[18,[-2]]],[],[]],[-1,[[18,[-2]]],[],[]],[-1,[[18,[-2]]],[],[]],[-1,[[18,[-2]]],[],[]],[-1,[[18,[-2]]],[],[]],[-1,[[18,[-2]]],[],[]],[-1,19,[]],[-1,19,[]],[-1,19,[]],[[[3,[4]],-1],-2,21,[]],[[[1,[4]],-1],-2,21,[]],[[[1,[4]],-1],-2,21,[]],[[[58,[-1]],-1],6,15],[[[58,[-1]],-1],6,15],0,0,[[[62,[-1]]],[[28,[4]]],50],[[[62,[-1]]],[[29,[4]]],50],[-1,-2,[],[]],[-1,-2,[],[]],[-1,-2,[],[]],[-1,-2,[],[]],[[[63,[-1]]],[],50],[[[63,[-1]]],[],50],[-1,-1,[]],[-1,-1,[]],[-1,-2,[],[]],[-1,-2,[],[]],[[[63,[-1]]],[[7,[[62,[-1]]]]],50],[[[62,[-1]]],[[7,[[63,[-1]]]]],50],[-1,[[7,[[63,[-1]]]]],50],[-1,[[18,[-2]]],[],[]],[-1,[[18,[-2]]],[],[]],[-1,[[18,[-2]]],[],[]],[-1,[[18,[-2]]],[],[]],[-1,19,[]],[-1,19,[]]],"c":[],"p":[[3,"DeviceMutRef",14],[8,"DeviceCopy",441],[3,"DeviceConstRef",14],[3,"DeviceAccessible",14],[3,"CombinedCudaAlloc",118],[15,"tuple"],[6,"CudaResult",442],[8,"CudaAlloc",118],[8,"Clone",443],[3,"Formatter",444],[6,"Result",444],[8,"Sized",445],[8,"Debug",444],[3,"SafeDeviceCopyWrapper",337],[8,"SafeDeviceCopy",264],[8,"TypeGraphLayout",446],[8,"CudaAsRust",14],[4,"Result",447],[3,"TypeId",448],[3,"ShallowCopy",61],[8,"FnOnce",449],[3,"PTXAllocator",77],[3,"Layout",450],[15,"u8"],[3,"Idx3",77],[3,"Dim3",77],[15,"usize"],[3,"HostAndDeviceMutRef",118],[3,"HostAndDeviceConstRef",118],[3,"LaunchConfig",118],[3,"TypedKernel",118],[15,"slice"],[4,"Option",451],[4,"KernelJITResult",118],[3,"HostDeviceBox",118],[3,"CudaDropWrapper",118],[15,"bool"],[3,"HostAndDeviceOwned",118],[15,"never"],[3,"DeviceBox",452],[3,"LaunchPackage",118],[4,"CudaError",442],[8,"From",453],[15,"str"],[3,"Function",454],[3,"SplitSliceOverCudaThreadsConstStride",273],[8,"AsMut",453],[3,"SplitSliceOverCudaThreadsDynamicStride",273],[8,"AsRef",453],[8,"RustToCuda",14],[8,"Borrow",455],[8,"BorrowMut",455],[8,"Deref",456],[8,"DerefMut",456],[3,"UnifiedAllocator",327],[3,"NonNull",457],[3,"AllocError",458],[3,"CudaExchangeItem",371],[3,"CudaExchangeBuffer",371],[3,"CudaExchangeBufferDevice",371],[3,"Vec",459],[3,"ExchangeWrapperOnDevice",418],[3,"ExchangeWrapperOnHost",418],[8,"RustToCudaProxy",14],[8,"BorrowFromRust",61],[8,"Launcher",118],[3,"NullCudaAlloc",118],[8,"LendToCuda",118]],"b":[[38,"impl-From%3C%26T%3E-for-DeviceAccessible%3CSafeDeviceCopyWrapper%3CT%3E%3E"],[40,"impl-From%3CT%3E-for-DeviceAccessible%3CT%3E"],[281,"impl-RustToCuda-for-SplitSliceOverCudaThreadsConstStride%3CT,+STRIDE%3E"],[282,"impl-Borrow%3C%5BE%5D%3E-for-SplitSliceOverCudaThreadsConstStride%3CT,+STRIDE%3E"],[284,"impl-Borrow%3C%5BE%5D%3E-for-SplitSliceOverCudaThreadsDynamicStride%3CT%3E"],[286,"impl-RustToCuda-for-SplitSliceOverCudaThreadsDynamicStride%3CT%3E"],[400,"impl-CudaExchangeItem%3CT,+M2D,+true%3E"],[401,"impl-CudaExchangeItem%3CT,+true,+M2H%3E"],[416,"impl-CudaExchangeItem%3CT,+true,+M2H%3E"],[417,"impl-CudaExchangeItem%3CT,+M2D,+true%3E"]]},\
"rust_cuda_derive":{"doc":"","t":"YX","n":["LendRustToCuda","kernel"],"q":[[0,"rust_cuda_derive"]],"d":["",""],"i":[0,0],"f":[0,0],"c":[],"p":[],"b":[]},\
"rust_cuda_ptx_jit":{"doc":"","t":"NDDENLLLLLLLLLLLLLLLLLLLLLLLLLL","n":["Cached","CudaKernel","PtxJITCompiler","PtxJITResult","Recomputed","borrow","borrow","borrow","borrow_mut","borrow_mut","borrow_mut","drop","from","from","from","get_function","into","into","into","new","new","try_from","try_from","try_from","try_into","try_into","try_into","type_id","type_id","type_id","with_arguments"],"q":[[0,"rust_cuda_ptx_jit"],[31,"rustacuda::function"],[32,"core::ffi::c_str"],[33,"rustacuda::error"],[34,"core::result"],[35,"core::any"],[36,"core::option"]],"d":["","","","","","","","","","","","","Returns the argument unchanged.","Returns the argument unchanged.","Returns the argument unchanged.","","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","","Errors","","","","","","","","","",""],"i":[12,0,0,0,12,5,12,1,5,12,1,1,5,12,1,1,5,12,1,5,1,5,12,1,5,12,1,5,12,1,5],"f":[0,0,0,0,0,[-1,-2,[],[]],[-1,-2,[],[]],[-1,-2,[],[]],[-1,-2,[],[]],[-1,-2,[],[]],[-1,-2,[],[]],[1,2],[-1,-1,[]],[-1,-1,[]],[-1,-1,[]],[1,3],[-1,-2,[],[]],[-1,-2,[],[]],[-1,-2,[],[]],[4,5],[[4,4],[[6,[1]]]],[-1,[[7,[-2]]],[],[]],[-1,[[7,[-2]]],[],[]],[-1,[[7,[-2]]],[],[]],[-1,[[7,[-2]]],[],[]],[-1,[[7,[-2]]],[],[]],[-1,[[7,[-2]]],[],[]],[-1,8,[]],[-1,8,[]],[-1,8,[]],[[5,[11,[[10,[[11,[[10,[9]]]]]]]]],12]],"c":[],"p":[[3,"CudaKernel",0],[15,"tuple"],[3,"Function",31],[3,"CStr",32],[3,"PtxJITCompiler",0],[6,"CudaResult",33],[4,"Result",34],[3,"TypeId",35],[15,"u8"],[15,"slice"],[4,"Option",36],[4,"PtxJITResult",0]],"b":[]}\
}');
if (typeof window !== 'undefined' && window.initSearch) {window.initSearch(searchIndex)};
if (typeof exports !== 'undefined') {exports.searchIndex = searchIndex};
