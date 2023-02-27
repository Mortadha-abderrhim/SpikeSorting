`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 12/12/2021 05:20:23 PM
// Design Name: 
// Module Name: tb_top_design
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module top_design_tb(

    );
    
     reg clk;
    reg rst_n;
    reg [1:0] ch_sel;
    //reg start;
    reg [7-1:0] data;
    //wire done;
    wire [14:0] result_a;
    wire [8:0] result_d;
    wire [10:0] result_s;
    
    top_design top_design_i (
        .clk(clk),
        .rst_n(rst_n),
        .ch_sel(ch_sel),
        //.start(start),
        .data(data),
        //.done(done),
        .result_a(result_a),
        .result_d(result_d),
        .result_s(result_s)
        );
        
    parameter PERIOD = 10;
    //reg [6:0] testdata [0:1279];
  initial clk = 1'b1;
  always #(PERIOD/2.0) clk = !clk;
  integer i;
  initial begin
    #(PERIOD) rst_n  =  1;data = 0;ch_sel = 0;
    #(PERIOD) rst_n  =  0;data = 0;ch_sel = 0;
    #(PERIOD) rst_n  =  1;data = 0;ch_sel = 0;
    #1;
    data = 60; ch_sel = 0;
	# (PERIOD)
	data = 50; ch_sel = 1;
	# (PERIOD)
	data = 55; ch_sel = 0;
	# (PERIOD)
	data = 40; ch_sel = 1;
	# (PERIOD)
	data = -30;ch_sel = 0;
	# (PERIOD)
	data = -20;ch_sel = 1;
	# (PERIOD)
	data = -23;ch_sel = 0;
	# (PERIOD)
	data = -25;ch_sel = 1;
	# (PERIOD)
	
    data = -3;ch_sel = 0;
	# (PERIOD)
    data = -5;ch_sel = 1;
	# (PERIOD)
    data = 6;ch_sel = 0;
	# (PERIOD)
    data = -20;ch_sel = 1;
	# (PERIOD)
    data = 0;ch_sel = 0; //x0_ch4
    # (PERIOD)
    data = 0;ch_sel = 1; //x0_ch4
    # (PERIOD)
    data= 0; ch_sel = 0; //x0_ch4
    # (PERIOD)
    data = 0; ch_sel = 1; //x0_ch4
    data = 7;ch_sel = 0;
	# (PERIOD)
    data = -15;ch_sel = 0;
    #50
    $finish;
end
endmodule
