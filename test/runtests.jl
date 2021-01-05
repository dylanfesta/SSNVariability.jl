using SSNVariability ; const S=SSNVariability
using Test

@testset "SSNVariability.jl" begin
    g1=S.ReLu(1.2)
    @test g1(-3.3) == 0
    @test g1(3.3) ≈ 3.3*1.2
    g2=S.ReQuad(4.4)
    @test g1(-3.3) == 0
    @test g1(3.3) ≈ 3.3*3.3*4.4
end
