var searchIndex = new Map(JSON.parse('[\
["rust_cuda",{"t":"QQQCCCQQDDDDCCRKRFFFYRKKNNNMMNNNNNNNNNNNNNNMMMNNNXMNNNNNNNNNNKFNNNNNNNNNNCMMMFFFNNHHNNNNNNNNNNNNHHNNNNHNNNNNNNNNOOOOOOPFRKFKFFFFGRFFKKFPFNNNONNNNNNNNNNNNNNNNNNNNNNNNNNNONNNNNNNNNNNNNNNNNNNNNNNNNNNNNMONNNNNNNNNNNOMMMNNNNNOONONNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNONNNKKKKKCCCCFFNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNFNNNNNNNNNFNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNCCFFFFNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNFFNNNNNNNNNNNNNNNNNNNNN","n":["assert","assert_eq","assert_ne","common","device","host","print","println","ptx_jit","rustacuda","rustacuda_core","rustacuda_derive","safety","utils","CudaAllocation","CudaAsRust","CudaRepresentation","DeviceAccessible","DeviceConstRef","DeviceMutRef","LendRustToCuda","RustRepresentation","RustToCuda","RustToCudaProxy","as_mut","as_ref","as_ref","as_rust","borrow","borrow","borrow","borrow","borrow_mut","borrow_mut","borrow_mut","clone","clone_into","fmt","from","from","from","from","from","from_mut","from_ref","into","into","into","into","kernel","restore","to_owned","try_from","try_from","try_from","try_into","try_into","try_into","type_id","type_id","type_id","BorrowFromRust","ShallowCopy","borrow","borrow_mut","deref","deref_mut","fmt","from","into","try_from","try_into","type_id","utils","with_borrow_from_rust","with_borrow_from_rust_mut","with_moved_from_rust","Dim3","Idx3","PTXAllocator","alloc","as_id","block_dim","block_idx","borrow","borrow","borrow","borrow_mut","borrow_mut","borrow_mut","dealloc","fmt","fmt","from","from","from","grid_dim","index","into","into","into","size","thread_idx","try_from","try_from","try_from","try_into","try_into","try_into","type_id","type_id","type_id","x","x","y","y","z","z","Cached","CombinedCudaAlloc","CompilationWatcher","CudaAlloc","CudaDropWrapper","EmptyCudaAlloc","HostAndDeviceConstRef","HostAndDeviceMutRef","HostAndDeviceOwned","HostDeviceBox","KernelJITResult","KernelTraitObject","LaunchConfig","LaunchPackage","Launcher","LendToCuda","NullCudaAlloc","Recompiled","TypedKernel","as_mut","as_ref","as_ref","block","borrow","borrow","borrow","borrow","borrow","borrow","borrow","borrow","borrow","borrow","borrow","borrow_mut","borrow_mut","borrow_mut","borrow_mut","borrow_mut","borrow_mut","borrow_mut","borrow_mut","borrow_mut","borrow_mut","borrow_mut","clone","clone","clone_into","clone_into","compile_with_ptx_jit_args","config","copy_from","copy_to","deref","deref_mut","drop","drop","eq","equivalent","equivalent","fmt","for_device","for_device","for_device","for_host","for_host","for_host","from","from","from","from","from","from","from","from","from","from","from","from","from","get_launch_package","grid","into","into","into","into","into","into","into","into","into","into","into","kernel","lend_to_cuda","lend_to_cuda_mut","move_to_cuda","new","new","new","new","on_compile","ptx_jit","shared_memory_size","split","stream","to_owned","to_owned","try_from","try_from","try_from","try_from","try_from","try_from","try_from","try_from","try_from","try_from","try_from","try_into","try_into","try_into","try_into","try_into","try_into","try_into","try_into","try_into","try_into","try_into","type_id","type_id","type_id","type_id","type_id","type_id","type_id","type_id","type_id","type_id","type_id","watcher","with_new","with_new","with_new","FitsIntoDeviceRegister","NoAliasing","SafeDeviceCopy","StackOnly","UnifiedHeapOnly","aliasing","alloc","device_copy","exchange","SplitSliceOverCudaThreadsConstStride","SplitSliceOverCudaThreadsDynamicStride","as_mut","as_mut","as_ref","as_ref","as_rust","as_rust","borrow","borrow","borrow","borrow","borrow","borrow","borrow_mut","borrow_mut","borrow_mut","borrow_mut","clone","clone","clone_into","clone_into","deref","deref","deref_mut","deref_mut","from","from","into","into","lend_to_cuda","lend_to_cuda","lend_to_cuda_mut","lend_to_cuda_mut","move_to_cuda","move_to_cuda","new","new","restore","restore","to_owned","to_owned","try_from","try_from","try_into","try_into","type_id","type_id","with_borrow_from_rust","with_borrow_from_rust","with_borrow_from_rust_mut","with_borrow_from_rust_mut","with_moved_from_rust","with_moved_from_rust","UnifiedAllocator","allocate","borrow","borrow_mut","deallocate","from","into","try_from","try_into","type_id","SafeDeviceCopyWrapper","as_rust","borrow","borrow","borrow_mut","clone","clone_into","fmt","from","from","from","from_mut","from_mut_slice","from_ref","from_slice","into","into_inner","into_mut","into_mut_slice","into_ref","into_slice","lend_to_cuda","lend_to_cuda_mut","move_to_cuda","restore","to_owned","try_from","try_into","type_id","with_borrow_from_rust","with_borrow_from_rust_mut","with_moved_from_rust","buffer","wrapper","CudaExchangeBuffer","CudaExchangeBufferDevice","CudaExchangeBufferHost","CudaExchangeItem","as_mut","borrow","borrow","borrow","borrow","borrow_mut","borrow_mut","borrow_mut","clone","clone_into","deref","deref","deref_mut","deref_mut","from","from","from","from_vec","into","into","into","lend_to_cuda","lend_to_cuda_mut","move_to_cuda","new","read","read","restore","to_owned","try_from","try_from","try_from","try_into","try_into","try_into","type_id","type_id","type_id","with_borrow_from_rust","with_borrow_from_rust_mut","with_moved_from_rust","write","write","ExchangeWrapperOnDevice","ExchangeWrapperOnHost","as_mut","as_ref","borrow","borrow","borrow_mut","borrow_mut","deref","deref_mut","from","from","into","into","move_to_device","move_to_host","new","try_from","try_from","try_into","try_into","type_id","type_id"],"q":[[0,"rust_cuda"],[14,"rust_cuda::common"],[61,"rust_cuda::device"],[77,"rust_cuda::device::utils"],[118,"rust_cuda::host"],[263,"rust_cuda::safety"],[268,"rust_cuda::utils"],[272,"rust_cuda::utils::aliasing"],[326,"rust_cuda::utils::alloc"],[336,"rust_cuda::utils::device_copy"],[368,"rust_cuda::utils::exchange"],[370,"rust_cuda::utils::exchange::buffer"],[417,"rust_cuda::utils::exchange::wrapper"],[440,"rustacuda_core::memory"],[441,"rustacuda::error"],[442,"const_type_layout"],[443,"core::clone"],[444,"core::fmt"],[445,"core::marker"],[446,"rust_cuda::safety::device_copy"],[447,"core::result"],[448,"core::any"],[449,"core::ops::function"],[450,"core::alloc::layout"],[451,"core::option"],[452,"rustacuda::memory::device::device_box"],[453,"core::convert"],[454,"rustacuda::function"],[455,"rust_cuda::utils::aliasing::const"],[456,"rust_cuda::utils::aliasing::dynamic"],[457,"core::borrow"],[458,"core::ops::deref"],[459,"core::ptr::non_null"],[460,"core::alloc"],[461,"rust_cuda::utils::exchange::buffer::device"],[462,"alloc::vec"],[463,"rust_cuda_derive"],[464,"rust_cuda::safety::register_fit"],[465,"rust_cuda::safety::no_aliasing"],[466,"rust_cuda::safety::stack_only"],[467,"rust_cuda::safety::unified_heap"],[468,"rust_cuda::utils::exchange::buffer::host"]],"i":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,10,0,10,0,0,0,0,7,0,0,2,5,2,7,10,8,5,2,8,5,2,5,5,8,8,8,8,5,2,25,25,25,8,5,2,0,10,5,8,5,2,8,5,2,8,5,2,0,0,28,28,28,28,28,28,28,28,28,28,0,29,29,29,0,0,0,32,35,0,0,32,36,35,32,36,35,32,36,35,32,36,35,0,0,32,36,35,36,0,32,36,35,32,36,35,32,36,35,36,35,36,35,36,35,44,0,52,0,0,0,0,0,0,0,0,52,0,0,0,0,0,44,0,38,38,39,40,53,44,41,79,12,46,45,38,39,48,40,53,44,41,79,12,46,45,38,39,48,40,39,40,39,40,41,53,45,45,46,46,46,45,40,40,40,40,38,39,48,38,39,48,53,44,41,79,12,46,46,45,45,38,39,48,40,52,40,53,44,41,79,12,46,45,38,39,48,40,53,54,54,54,41,12,38,39,52,40,40,12,53,39,40,53,44,41,79,12,46,45,38,39,48,40,53,44,41,79,12,46,45,38,39,48,40,53,44,41,79,12,46,45,38,39,48,40,53,38,39,48,0,0,0,0,0,0,0,0,0,0,0,59,61,59,61,59,61,59,59,59,61,61,61,59,59,61,61,59,61,59,61,59,61,59,61,59,61,59,61,59,61,59,61,59,61,59,61,59,61,59,61,59,61,59,61,59,61,59,61,59,61,59,61,0,68,68,68,68,68,68,68,68,68,0,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,0,0,0,0,0,0,72,74,73,73,72,74,73,72,72,72,74,73,74,73,74,73,72,73,74,73,72,73,73,73,73,72,72,73,72,74,73,72,74,73,72,74,73,72,73,73,73,72,72,0,0,76,76,78,76,78,76,78,78,78,76,78,76,78,76,78,78,76,78,76,78,76],"f":"````````````````````````{{{f{b{d{c}}}}}{{f{bc}}}h}{{{f{{j{c}}}}}{{f{c}}}h}{{{f{{d{c}}}}}{{f{c}}}h}{{{f{{A`{{n{}{{l{c}}}}}}}}}c{{Ad{}{{Ab{}}}}}}{{{f{{Ad{}{{Af{c}}{Ab{e}}}}}}g}{{Al{{Aj{{A`{e}}{Ah{cg}}}}}}}An{{n{}{{l{}}}}B`}An}{{{f{c}}}{{f{e}}}{}{}}00{{{f{bc}}}{{f{be}}}{}{}}00{{{f{{j{c}}}}}{{j{c}}}{Bbh}}{{{f{c}}{f{be}}}Bd{}{}}{{{f{{A`{c}}}}{f{bBf}}}Bh{BjhBl}}{cc{}}{c{{A`{c}}}n}{{{f{c}}}{{A`{{Bn{c}}}}}{C`B`}}22{{{f{bc}}}{{f{bCb}}}{}}{{{f{c}}}{{f{Cb}}}{}}{Cbc{}}{ce{}{}}00`{{{f{b{Ad{}{{Af{c}}{Ab{e}}}}}}{Ah{cg}}}{{Al{g}}}An{{n{}{{l{}}}}B`}An}{{{f{c}}}e{}{}}{c{{Cd{e}}}{}{}}00000{{{f{c}}}Cf{}}00``?>{{{f{{Ch{c}}}}}{{f{e}}}{}{}}{{{f{b{Ch{c}}}}}{{f{be}}}{}{}}{{{f{{Ch{c}}}}{f{bBf}}}BhBl}=7443`{{{j{{A`{c}}}}g}e{}{}{{Cn{{f{{Ch{Cj}}}}}{{Cl{e}}}}}}{{{d{{A`{c}}}}g}e{}{}{{Cn{{f{b{Ch{Cj}}}}}{{Cl{e}}}}}}{{{d{{A`{c}}}}g}e{}{}{{Cn{Cj}{{Cl{e}}}}}}```{{{f{D`}}Db}Dd}{{{f{Df}}{f{Dh}}}Dj}{{}Dh}{{}Df}{{{f{c}}}{{f{e}}}{}{}}00{{{f{bc}}}{{f{be}}}{}{}}00{{{f{D`}}DdDb}Bd}{{{f{Dh}}{f{bBf}}}Bh}{{{f{Df}}{f{bBf}}}Bh}{cc{}}007{{}Dj}{ce{}{}}00{{{f{Dh}}}Dj}9{c{{Cd{e}}}{}{}}00000{{{f{c}}}Cf{}}00`````````````````````````{{{f{b{Dl{c}}}}}{{Dl{c}}}h}{{{f{{Dl{c}}}}}{{Dn{c}}}h}{{{f{{Dn{c}}}}}{{Dn{c}}}h}`===========<<<<<<<<<<<0{{{f{E`}}}E`}{{{f{c}}{f{be}}}Bd{}{}}0{{{f{b{Eb{c}}}}{Ef{{f{{Ed{{Ef{{f{{Ed{Dd}}}}}}}}}}}}}{{Al{Eh}}}Bj}`{{{f{b{Ej{c}}}}{f{c}}}{{Al{Bd}}}h}{{{f{{Ej{c}}}}{f{bc}}}{{Al{Bd}}}h}{{{f{{El{c}}}}}{{f{e}}}`{}}{{{f{b{El{c}}}}}{{f{be}}}`{}}{{{f{b{El{c}}}}}Bd`}{{{f{b{Ej{c}}}}}Bdh}{{{f{E`}}{f{E`}}}En}{{{f{c}}{f{e}}}En{}{}}0{{{f{E`}}{f{bBf}}}Bh}{{{f{b{Dl{c}}}}}{{d{c}}}h}{{{f{{Dn{c}}}}}{{j{c}}}h}{{{F`{c}}}{{d{c}}}{C`h}}{{{f{{Dl{c}}}}}{{f{c}}}h}{{{f{{Dn{c}}}}}{{f{c}}}h}{{{f{b{F`{c}}}}}{{f{c}}}{C`h}}{cc{}}00000{c{{El{c}}}`}1{{{Fb{c}}}{{Ej{c}}}h}2222{{{f{b{Fh{}{{Fd{c}}{Ff{e}}}}}}}{{Fj{{Fh{}{{Fd{c}}{Ff{e}}}}}}}Bj{}}`{ce{}{}}0000000000`{{{f{Fl}}i}{{Cd{eg}}}{}{}{{G`{Fn}}}{{Cn{{Dn{{A`{c}}}}}{{Cl{{Cd{eg}}}}}}}}{{{f{bFl}}i}{{Cd{eg}}}{}{}{{G`{Fn}}}{{Cn{{Dl{{A`{c}}}}}{{Cl{{Cd{eg}}}}}}}}{{Fli}{{Cd{eg}}}{}{}{{G`{Fn}}}{{Cn{{F`{{A`{c}}}}}{{Cl{{Cd{eg}}}}}}}}{{{f{Gb}}{f{Gb}}}{{Al{{Eb{c}}}}}Bj}{{ce}{{Ah{ce}}}AnAn}{{{f{b{Ej{c}}}}{f{bc}}}{{Dl{c}}}h}{{{f{{Ej{c}}}}{f{c}}}{{Dn{c}}}h}{{{f{Gd}}{f{bc}}}{{Al{Bd}}}{}}``{{{Ah{ce}}}{{Aj{ce}}}AnAn}`{{{f{c}}}e{}{}}0{c{{Cd{e}}}{}{}}000000000000000000000{{{f{c}}}Cf{}}0000000000`{{{f{bc}}i}{{Cd{eg}}}h{}{{G`{Fn}}}{{Cn{{Dl{c}}}{{Cl{{Cd{eg}}}}}}}}{{{f{c}}i}{{Cd{eg}}}h{}{{G`{Fn}}}{{Cn{{Dn{c}}}{{Cl{{Cd{eg}}}}}}}}{{ci}{{Cd{eg}}}{C`h}{}{{G`{Fn}}}{{Cn{{F`{c}}}{{Cl{{Cd{eg}}}}}}}}```````````{{{f{b{Gf{e}}}}}{{f{b{Ed{c}}}}}{}{{Gh{{Ed{c}}}}}}{{{f{b{Gj{e}}}}}{{f{b{Ed{c}}}}}{}{{Gh{{Ed{c}}}}}}{{{f{{Gf{e}}}}}{{f{{Ed{c}}}}}{}{{Gl{{Ed{c}}}}}}{{{f{{Gj{e}}}}}{{f{{Ed{c}}}}}{}{{Gl{{Ed{c}}}}}}{{{f{{A`{{Gf{{A`{c}}}}}}}}}en{}}{{{f{{A`{{Gj{{A`{c}}}}}}}}}en{}}{{{f{{Gf{c}}}}e}{{Al{{Aj{{A`{g}}{Ah{ie}}}}}}}AdAn{}{}}{{{f{{Gf{e}}}}}{{f{{Ed{c}}}}}{}{{Gn{{Ed{c}}}}}}{{{f{c}}}{{f{e}}}{}{}}0{{{f{{Gj{e}}}}}{{f{{Ed{c}}}}}{}{{Gn{{Ed{c}}}}}}{{{f{{Gj{c}}}}e}{{Al{{Aj{{A`{g}}{Ah{ie}}}}}}}AdAn{}{}}{{{f{bc}}}{{f{be}}}{}{}}{{{f{b{Gf{e}}}}}{{f{b{Ed{c}}}}}{}{{H`{{Ed{c}}}}}}{{{f{b{Gj{e}}}}}{{f{b{Ed{c}}}}}{}{{H`{{Ed{c}}}}}}2{{{f{{Gf{c}}}}}{{Gf{c}}}Bb}{{{f{{Gj{c}}}}}{{Gj{c}}}Bb}{{{f{c}}{f{be}}}Bd{}{}}0{{{f{{Gf{e}}}}}{{f{g}}}{}{{Hd{}{{Hb{{Ed{c}}}}}}}{}}{{{f{{Gj{e}}}}}{{f{g}}}{}{{Hd{}{{Hb{{Ed{c}}}}}}}{}}{{{f{b{Gf{e}}}}}{{f{bg}}}{}{{Hf{}{{Hb{{Ed{c}}}}}}}{}}{{{f{b{Gj{e}}}}}{{f{bg}}}{}{{Hf{}{{Hb{{Ed{c}}}}}}}{}}{cc{}}0{ce{}{}}0{{{f{c}}i}{{Cd{eg}}}{}{}{{G`{Fn}}}{{Cn{{Dn{A`}}}{{Cl{{Cd{eg}}}}}}}}0{{{f{bc}}i}{{Cd{eg}}}{}{}{{G`{Fn}}}{{Cn{{Dl{A`}}}{{Cl{{Cd{eg}}}}}}}}0{{ci}{{Cd{eg}}}{}{}{{G`{Fn}}}{{Cn{{F`{A`}}}{{Cl{{Cd{eg}}}}}}}}0{c{{Gf{c}}}{}}{{cDj}{{Gj{c}}}{}}{{{f{b{Gf{c}}}}{Ah{eg}}}{{Al{g}}}Ad{}An}{{{f{b{Gj{c}}}}{Ah{eg}}}{{Al{g}}}Ad{}An}{{{f{c}}}e{}{}}0{c{{Cd{e}}}{}{}}000{{{f{c}}}Cf{}}0{{{j{A`}}g}e{}{}{{Cn{{f{{Ch{c}}}}}{{Cl{e}}}}}}0{{{d{A`}}g}e{}{}{{Cn{{f{b{Ch{c}}}}}{{Cl{e}}}}}}0{{{d{A`}}g}eC`{}{{Cn{c}{{Cl{e}}}}}}0`{{{f{Hh}}Db}{{Cd{{Hj{{Ed{Dd}}}}Hl}}}}{{{f{c}}}{{f{e}}}{}{}}{{{f{bc}}}{{f{be}}}{}{}}{{{f{Hh}}{Hj{Dd}}Db}Bd}{cc{}}{ce{}{}}::9`{{{f{{A`{{Bn{c}}}}}}}e{C`B`}{}}5{{{f{{Bn{c}}}}e}{{Al{{Aj{{A`{g}}{Ah{ie}}}}}}}{C`B`}An{}{}}5{{{f{{Bn{c}}}}}{{Bn{c}}}{C`B`Bb}}{{{f{c}}{f{be}}}Bd{}{}}{{{f{{Bn{c}}}}{f{bBf}}}Bh{C`B`Bl}}{Hnc{}}{c{{Bn{c}}}{C`B`}}8{{{f{bc}}}{{f{b{Bn{c}}}}}{C`B`}}{{{f{b{Ed{c}}}}}{{f{b{Ed{{Bn{c}}}}}}}{C`B`}}{{{f{c}}}{{f{{Bn{c}}}}}{C`B`}}{{{f{{Ed{c}}}}}{{f{{Ed{{Bn{c}}}}}}}{C`B`}};{{{Bn{c}}}c{C`B`}}{{{f{b{Bn{c}}}}}{{f{bc}}}{C`B`}}{{{f{b{Ed{{Bn{c}}}}}}}{{f{b{Ed{c}}}}}{C`B`}}{{{f{{Bn{c}}}}}{{f{c}}}{C`B`}}{{{f{{Ed{{Bn{c}}}}}}}{{f{{Ed{c}}}}}{C`B`}}{{{f{c}}i}{{Cd{eg}}}{}{}{{G`{Fn}}}{{Cn{{Dn{A`}}}{{Cl{{Cd{eg}}}}}}}}{{{f{bc}}i}{{Cd{eg}}}{}{}{{G`{Fn}}}{{Cn{{Dl{A`}}}{{Cl{{Cd{eg}}}}}}}}{{ci}{{Cd{eg}}}{}{}{{G`{Fn}}}{{Cn{{F`{A`}}}{{Cl{{Cd{eg}}}}}}}}{{{f{b{Bn{c}}}}{Ah{eg}}}{{Al{g}}}{C`B`}{}An}{{{f{c}}}e{}{}}{c{{Cd{e}}}{}{}}0{{{f{c}}}Cf{}}{{{j{A`}}g}e{}{}{{Cn{{f{{Ch{c}}}}}{{Cl{e}}}}}}{{{d{A`}}g}e{}{}{{Cn{{f{b{Ch{c}}}}}{{Cl{e}}}}}}{{{d{A`}}g}eC`{}{{Cn{c}{{Cl{e}}}}}}``````{{{f{b{I`{c}}}}}{{f{bc}}}C`}{{{f{c}}}{{f{e}}}{}{}}{{{f{{Ib{c}}}}e}{{Al{{Aj{{A`{g}}{Ah{ie}}}}}}}{C`B`}An{}{}}11{{{f{bc}}}{{f{be}}}{}{}}00{{{f{{I`{c}}}}}{{I`{c}}}{BbC`}}{{{f{c}}{f{be}}}Bd{}{}}{{{f{{Id{c}}}}}{{f{e}}}C`{}}{{{f{{Ib{c}}}}}{{f{e}}}{C`B`}{}}{{{f{b{Id{c}}}}}{{f{be}}}C`{}}{{{f{b{Ib{c}}}}}{{f{be}}}{C`B`}{}}{cc{}}00{{{If{c}}}{{Al{{Ib{c}}}}}{C`B`}}{ce{}{}}00{{{f{c}}i}{{Cd{eg}}}{}{}{{G`{Fn}}}{{Cn{{Dn{A`}}}{{Cl{{Cd{eg}}}}}}}}{{{f{bc}}i}{{Cd{eg}}}{}{}{{G`{Fn}}}{{Cn{{Dl{A`}}}{{Cl{{Cd{eg}}}}}}}}{{ci}{{Cd{eg}}}{}{}{{G`{Fn}}}{{Cn{{F`{A`}}}{{Cl{{Cd{eg}}}}}}}}{{{f{c}}Dj}{{Al{{Ib{c}}}}}{BbC`B`}}{{{f{{I`{c}}}}}{{f{c}}}C`}0{{{f{b{Ib{c}}}}{Ah{eg}}}{{Al{g}}}{C`B`}{}An}{{{f{c}}}e{}{}}{c{{Cd{e}}}{}{}}00000{{{f{c}}}Cf{}}00{{{j{A`}}g}e{}{}{{Cn{{f{{Ch{c}}}}}{{Cl{e}}}}}}{{{d{A`}}g}e{}{}{{Cn{{f{b{Ch{c}}}}}{{Cl{e}}}}}}{{{d{A`}}g}eC`{}{{Cn{c}{{Cl{e}}}}}}{{{f{b{I`{c}}}}c}BdC`}0``{{{f{b{Ih{c}}}}}{{Dl{A`}}}{{Ad{}{{Af{Ij}}}}}}{{{f{{Ih{c}}}}}{{Dn{A`}}}{{Ad{}{{Af{Ij}}}}}}{{{f{c}}}{{f{e}}}{}{}}0{{{f{bc}}}{{f{be}}}{}{}}0{{{f{{Il{c}}}}}{{f{e}}}{{Ad{}{{Af{Ij}}}}}{}}{{{f{b{Il{c}}}}}{{f{be}}}{{Ad{}{{Af{Ij}}}}}{}}{cc{}}0{ce{}{}}0{{{Il{c}}}{{Al{{Ih{c}}}}}{{Ad{}{{Af{Ij}}}}}}{{{Ih{c}}}{{Al{{Il{c}}}}}{{Ad{}{{Af{Ij}}}}}}{c{{Al{{Il{c}}}}}{{Ad{}{{Af{Ij}}}}}}{c{{Cd{e}}}{}{}}000{{{f{c}}}Cf{}}0","D":"Kf","p":[[0,"mut"],[5,"DeviceMutRef",14],[1,"reference"],[10,"DeviceCopy",440],[5,"DeviceConstRef",14],[17,"RustRepresentation"],[10,"CudaAsRust",14],[5,"DeviceAccessible",14],[17,"CudaRepresentation"],[10,"RustToCuda",14],[17,"CudaAllocation"],[5,"CombinedCudaAlloc",118],[1,"tuple"],[8,"CudaResult",441],[10,"CudaAlloc",118],[10,"TypeGraphLayout",442],[10,"Clone",443],[1,"unit"],[5,"Formatter",444],[8,"Result",444],[10,"Sized",445],[10,"Debug",444],[5,"SafeDeviceCopyWrapper",336],[10,"SafeDeviceCopy",263,446],[10,"RustToCudaProxy",14],[6,"Result",447],[5,"TypeId",448],[5,"ShallowCopy",61],[10,"BorrowFromRust",61],[17,"Output"],[10,"FnOnce",449],[5,"PTXAllocator",77],[5,"Layout",450],[1,"u8"],[5,"Idx3",77],[5,"Dim3",77],[1,"usize"],[5,"HostAndDeviceMutRef",118],[5,"HostAndDeviceConstRef",118],[5,"LaunchConfig",118],[5,"TypedKernel",118],[1,"slice"],[6,"Option",451],[6,"KernelJITResult",118],[5,"HostDeviceBox",118],[5,"CudaDropWrapper",118],[1,"bool"],[5,"HostAndDeviceOwned",118],[5,"DeviceBox",452],[17,"KernelTraitObject"],[17,"CompilationWatcher"],[10,"Launcher",118],[5,"LaunchPackage",118],[10,"LendToCuda",118],[6,"CudaError",441],[10,"From",453],[1,"str"],[5,"Function",454],[5,"SplitSliceOverCudaThreadsConstStride",272,455],[10,"AsMut",453],[5,"SplitSliceOverCudaThreadsDynamicStride",272,456],[10,"AsRef",453],[10,"Borrow",457],[10,"BorrowMut",457],[17,"Target"],[10,"Deref",458],[10,"DerefMut",458],[5,"UnifiedAllocator",326],[5,"NonNull",459],[5,"AllocError",460],[1,"never"],[5,"CudaExchangeItem",370],[5,"CudaExchangeBuffer",370],[5,"CudaExchangeBufferDevice",370,461],[5,"Vec",462],[5,"ExchangeWrapperOnDevice",417],[10,"EmptyCudaAlloc",118],[5,"ExchangeWrapperOnHost",417],[5,"NullCudaAlloc",118]],"r":[[20,463],[49,463],[263,464],[264,465],[265,446],[266,466],[267,467],[272,455],[273,456],[371,461],[372,468]],"b":[[39,"impl-From%3CT%3E-for-DeviceAccessible%3CT%3E"],[40,"impl-From%3C%26T%3E-for-DeviceAccessible%3CSafeDeviceCopyWrapper%3CT%3E%3E"],[280,"impl-RustToCuda-for-SplitSliceOverCudaThreadsConstStride%3CT,+STRIDE%3E"],[281,"impl-Borrow%3C%5BE%5D%3E-for-SplitSliceOverCudaThreadsConstStride%3CT,+STRIDE%3E"],[284,"impl-Borrow%3C%5BE%5D%3E-for-SplitSliceOverCudaThreadsDynamicStride%3CT%3E"],[285,"impl-RustToCuda-for-SplitSliceOverCudaThreadsDynamicStride%3CT%3E"],[399,"impl-CudaExchangeItem%3CT,+M2D,+true%3E"],[400,"impl-CudaExchangeItem%3CT,+true,+M2H%3E"],[415,"impl-CudaExchangeItem%3CT,+M2D,+true%3E"],[416,"impl-CudaExchangeItem%3CT,+true,+M2H%3E"]],"c":"OjAAAAAAAAA=","e":"OzAAAAEAAF0BHwAAAAAABAACAAkABgARAAUAGAADAB4ACAAoAAEALAACADIAAAA0ABAARwADAFEADABhAAEAZgBBAKkAAACsAA0AwAAAAMIAAADHAAEA1AAAANkAAADdACcACAEiAC8BHABOAQwAXAEDAGEBEQB2AQ4AjAECAJABGwCzAQUA"}],\
["rust_cuda_derive",{"t":"YX","n":["LendRustToCuda","kernel"],"q":[[0,"rust_cuda_derive"]],"i":[0,0],"f":"``","D":"`","p":[],"r":[],"b":[],"c":"OjAAAAAAAAA=","e":"OjAAAAEAAAAAAAIAEAAAAAAAAQACAA=="}],\
["rust_cuda_ptx_jit",{"t":"PFFGPNNNNNNNNNNNNNNNNNNNNNNNNNN","n":["Cached","CudaKernel","PtxJITCompiler","PtxJITResult","Recomputed","borrow","borrow","borrow","borrow_mut","borrow_mut","borrow_mut","drop","from","from","from","get_function","into","into","into","new","new","try_from","try_from","try_from","try_into","try_into","try_into","type_id","type_id","type_id","with_arguments"],"q":[[0,"rust_cuda_ptx_jit"],[31,"rust_cuda_ptx_jit::host::kernel"],[32,"rustacuda::function"],[33,"core::ffi::c_str"],[34,"rust_cuda_ptx_jit::host::compiler"],[35,"rustacuda::error"],[36,"core::result"],[37,"core::any"],[38,"core::option"]],"i":[14,0,0,0,14,7,14,3,7,14,3,3,7,14,3,3,7,14,3,7,3,7,14,3,7,14,3,7,14,3,7],"f":"`````{{{b{c}}}{{b{e}}}{}{}}00{{{b{dc}}}{{b{de}}}{}{}}00{{{b{df}}}h}{cc{}}00{{{b{f}}}{{b{j}}}}{ce{}{}}00{{{b{l}}}n}{{{b{l}}{b{l}}}{{A`{f}}}}{c{{Ab{e}}}{}{}}00000{{{b{c}}}Ad{}}00{{{b{dn}}{Aj{{b{{Ah{{Aj{{b{{Ah{Af}}}}}}}}}}}}}Al}","D":"n","p":[[1,"reference"],[0,"mut"],[5,"CudaKernel",0,31],[1,"unit"],[5,"Function",32],[5,"CStr",33],[5,"PtxJITCompiler",0,34],[8,"CudaResult",35],[6,"Result",36],[5,"TypeId",37],[1,"u8"],[1,"slice"],[6,"Option",38],[6,"PtxJITResult",0,34]],"r":[[1,31],[2,34],[3,34]],"b":[],"c":"OjAAAAAAAAA=","e":"OzAAAAEAABgABAAAAAwAEAAAABQAAAAWAAkA"}]\
]'));
if (typeof exports !== 'undefined') exports.searchIndex = searchIndex;
else if (window.initSearch) window.initSearch(searchIndex);
