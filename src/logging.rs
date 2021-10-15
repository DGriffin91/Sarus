use std::fmt;
use tracing_core::{Event, Subscriber};
use tracing_subscriber::fmt::{FmtContext, FormatEvent, FormatFields, FormattedFields};
use tracing_subscriber::registry::LookupSpan;

use tracing::info;
use tracing_subscriber::{
    filter, prelude::__tracing_subscriber_SubscriberExt, util::SubscriberInitExt, Layer,
};

struct MyFormatter;

impl<S, N> FormatEvent<S, N> for MyFormatter
where
    S: Subscriber + for<'a> LookupSpan<'a>,
    N: for<'a> FormatFields<'a> + 'static,
{
    fn format_event(
        &self,
        ctx: &FmtContext<'_, S, N>,
        writer: &mut dyn fmt::Write,
        event: &Event<'_>,
    ) -> fmt::Result {
        let meta = event.metadata();

        write!(
            writer,
            "{}:{}\t{}\t{}:",
            meta.file().unwrap_or(""),
            meta.line().unwrap_or(0),
            meta.level(),
            meta.target(),
        )?;

        let mut last_span_name = "";
        let mut rep_span_counter = 0usize;
        let mut span_counter = 0usize;

        // Write spans and fields of each span
        ctx.visit_spans(|span| {
            if last_span_name == span.name() {
                rep_span_counter += 1;
            } else {
                if rep_span_counter == 0 && span_counter > 0 {
                    write!(writer, ":")?;
                }
                if rep_span_counter > 0 {
                    write!(writer, "(x{}):", rep_span_counter + 1)?;
                }
                write!(writer, "{}", span.name())?;
                rep_span_counter = 0;
            }
            last_span_name = span.name();
            span_counter += 1;

            let ext = span.extensions();

            // `FormattedFields` is a a formatted representation of the span's
            // fields, which is stored in its extensions by the `fmt` layer's
            // `new_span` method. The fields will have been formatted
            // by the same field formatter that's provided to the event
            // formatter in the `FmtContext`.
            let fields = &ext
                .get::<FormattedFields<N>>()
                .expect("will never be `None`");

            if !fields.is_empty() {
                write!(writer, "{{{}}}", fields)?;
            }

            Ok(())
        })?;

        write!(writer, ": ")?;

        // Write fields on the event
        ctx.field_format().format_fields(writer, event)?;

        writeln!(writer)
    }
}

pub fn setup_logging() {
    // Some spans:
    //sarus::jit
    //cranelift_jit::backend

    // install global collector configured based on RUST_LOG env var.
    let my_filter = filter::filter_fn(|metadata| metadata.target().contains("sarus"));

    let layer = tracing_subscriber::fmt::layer().event_format(MyFormatter);

    tracing_subscriber::registry()
        .with(layer.with_filter(my_filter))
        .init();

    /*

    // write json to file
    let file_appender =
        tracing_appender::rolling::hourly("C:\\Users\\DGriffin\\tmp\\", "sarus.json");
    let (non_blocking, guard) = tracing_appender::non_blocking(file_appender);

    let layer = tracing_subscriber::fmt::layer()
        .json()
        .with_writer(non_blocking);

    tracing_subscriber::registry()
        .with(layer.with_filter(my_filter))
        .init();

    return guard; //This style needs this guard to outlive the Sarus compilation

    */

    /*

    // use Etw for PrefView
    Also need tracing-etw = "0.1.0"
    tracing::subscriber::set_global_default(
        tracing_subscriber::registry().with(tracing_etw::EtwLayer::new(true)),
    )
    .expect("setup the subscriber");

    */

    info!("setup_logging");
}
