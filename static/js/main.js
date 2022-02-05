/*global $ */
$(function () {
    
    "use strict";
    $('.skill-progress span').each(function () {

        $(this).animate({'width': $(this).data('width')}, 1000);
    });
});